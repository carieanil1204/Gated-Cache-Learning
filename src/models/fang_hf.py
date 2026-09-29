"""Hugging Face causal LM backend for the Fang GAPG estimator.

Rollouts are sampled from the current policy at temperature 1 without top-k
or top-p filtering. Each batch gets exactly one optimizer step. The score
is the sum of log probabilities of local completion tokens, including EOS;
prompt and cloud tokens never contribute. Dropout is disabled in both passes.
"""

import json
from pathlib import Path
import time

from src.models.fang import FangGroup, FangRollout, UNKNOWN_MARKER, group_coefficients


# A paraphrase of the paper's Template I, not its verbatim prompt.
SYSTEM_PROMPT = (
    "Work through the task, enclosing your reasoning in <think>...</think>. "
    "End with your solution inside <answer>...</answer>. "
    "If you cannot solve it confidently, finish with " + UNKNOWN_MARKER
)
LOCAL_ONLY_PROMPT = (
    "Solve the task yourself. External assistance is unavailable. "
    "End with your best solution inside <answer>...</answer>."
)


def sequence_log_probability(model, rollout: FangRollout):
    """Differentiable completion likelihood, with the causal shift applied."""
    import torch
    import torch.nn.functional as functional

    if not rollout.prompt_ids or not rollout.completion_ids:
        raise ValueError("likelihood requires the original prompt and completion token IDs")
    device = next(model.parameters()).device
    tokens = torch.tensor([rollout.prompt_ids + rollout.completion_ids], device=device)
    logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
    start = len(rollout.prompt_ids) - 1
    completion_logits = logits[:, start:-1, :].float()
    targets = tokens[:, len(rollout.prompt_ids):]
    return -functional.cross_entropy(
        completion_logits.reshape(-1, completion_logits.shape[-1]),
        targets.reshape(-1), reduction="sum",
    )


class HuggingFaceFangPolicy:
    def __init__(
        self, model, tokenizer, *, learning_rate: float = 5e-6,
        max_prompt_tokens: int = 256, max_new_tokens: int = 720,
        metadata: dict | None = None,
    ):
        import torch

        if learning_rate <= 0 or max_prompt_tokens < 1 or max_new_tokens < 1:
            raise ValueError("learning rate and token limits must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.metadata = metadata or {}
        self.model.eval()  # eval disables dropout, but still permits autograd
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        # SGD implements the paper's stated update directly, without weight decay.
        self.optimizer = torch.optim.SGD(trainable, lr=learning_rate) if trainable else None

    @classmethod
    def from_pretrained(cls, settings: dict, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run outside the device-restricted sandbox or select --device cpu")
        name = settings["model_name"]
        revision = settings.get("revision")
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, trust_remote_code=False,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        ).to(device)
        rank = settings.get("lora_rank", 0)
        if rank:
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(model, LoraConfig(
                task_type="CAUSAL_LM", r=rank, lora_alpha=settings["lora_alpha"],
                lora_dropout=0.0, target_modules=settings["lora_target_modules"],
            ))
        metadata = dict(settings)
        metadata["revision"] = getattr(model.config, "_commit_hash", None) or revision
        return cls(
            model, tokenizer, learning_rate=settings["learning_rate"],
            max_prompt_tokens=settings["max_prompt_tokens"],
            max_new_tokens=settings["max_new_tokens"], metadata=metadata,
        )

    def _prompt_ids(self, prompt: str, allow_help: bool = True) -> tuple[int, ...]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT if allow_help else LOCAL_ONLY_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if self.tokenizer.chat_template:
            ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        else:
            text = messages[0]["content"] + "\nQuestion: " + prompt + "\nResponse:"
            ids = self.tokenizer.encode(text, add_special_tokens=True)
        if not ids or len(ids) > self.max_prompt_tokens:
            raise ValueError(f"prompt has {len(ids)} tokens; allowed 1..{self.max_prompt_tokens}; no silent truncation")
        return tuple(ids)

    def _generate(self, prompt_ids: tuple[int, ...], sample: bool) -> FangRollout:
        import torch
        from transformers import GenerationConfig

        device = next(self.model.parameters()).device
        tokens = torch.tensor([prompt_ids], device=device)
        # Build a fresh config so checkpoint-specific sampling processors cannot
        # silently change the behavior policy relative to likelihood scoring.
        generation = GenerationConfig(
            max_new_tokens=self.max_new_tokens, do_sample=sample,
            temperature=1.0, top_k=0 if sample else 50, top_p=1.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                          else self.tokenizer.eos_token_id),
            use_cache=True,
        )
        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(
                input_ids=tokens, attention_mask=torch.ones_like(tokens), generation_config=generation,
            )
        completion = tuple(output[0, len(prompt_ids):].tolist())
        text = self.tokenizer.decode(completion, skip_special_tokens=True)
        return FangRollout(text, prompt_ids, completion)

    def sample(self, prompt: str, count: int) -> list[FangRollout]:
        if count < 1:
            raise ValueError("sample count must be positive")
        ids = self._prompt_ids(prompt)
        # Sequential rollouts and backwards keep memory bounded on a 12 GB GPU.
        return [self._generate(ids, sample=True) for _ in range(count)]

    def answer(self, prompt: str, allow_help: bool = True) -> tuple[str, float]:
        start = time.perf_counter()
        rollout = self._generate(self._prompt_ids(prompt, allow_help), sample=False)
        return rollout.text, (time.perf_counter() - start) * 1000

    def update(self, groups: list[FangGroup]) -> float:
        import torch

        if not groups:
            raise ValueError("cannot update an empty batch")
        if self.optimizer is None:
            raise RuntimeError("policy has no trainable parameters")
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        active = False
        for group in groups:
            if len(group.responses) != len(group.rewards):
                raise ValueError("rollout and reward counts differ")
            for rollout, coefficient in zip(group.responses, group_coefficients(group.rewards)):
                if coefficient == 0:
                    continue
                loss = -sequence_log_probability(self.model, rollout) * coefficient / len(groups)
                if not torch.isfinite(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("nonfinite GAPG loss")
                loss.backward()
                total_loss += loss.detach().item()
                active = True
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if any(not torch.isfinite(p.grad).all() for p in parameters):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("nonfinite GAPG gradient")
        if active:
            self.optimizer.step()
        return total_loss

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        metadata = dict(self.metadata)
        metadata.update(max_prompt_tokens=self.max_prompt_tokens, max_new_tokens=self.max_new_tokens)
        (path / "fang_policy.json").write_text(json.dumps(metadata, indent=2) + "\n")

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(path)
        metadata = json.loads((path / "fang_policy.json").read_text())
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run outside the device-restricted sandbox or select --device cpu")
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        if (path / "adapter_config.json").exists():
            from peft import PeftModel

            base = AutoModelForCausalLM.from_pretrained(
                metadata["model_name"], revision=metadata.get("revision"),
                torch_dtype=dtype, trust_remote_code=False,
            ).to(device)
            model = PeftModel.from_pretrained(base, path, is_trainable=False)
        else:
            model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, trust_remote_code=False).to(device)
        model.requires_grad_(False)
        return cls(
            model, tokenizer, max_prompt_tokens=metadata["max_prompt_tokens"],
            max_new_tokens=metadata["max_new_tokens"], metadata=metadata,
        )
