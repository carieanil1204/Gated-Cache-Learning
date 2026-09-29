"""Numerical gradient and real Transformers integration checks, no downloads."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.models.fang import FangExample, FangGroup, FangRollout
from src.models.fang_hf import HuggingFaceFangPolicy, sequence_log_probability


class TwoTokenModel(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(2, device=device))

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, 2))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradient_matches_analytical_policy_gradient(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable in this process")
    model = TwoTokenModel(device)
    policy = HuggingFaceFangPolicy(model, None, learning_rate=0.1)
    result = FangGroup(
        FangExample("q", "a"),
        [FangRollout("a", (0, 1), (0,)), FangRollout("b", (0, 1), (1,))],
        [1.0, 0.0], [True, False], None,
    )
    policy.update([result])
    assert model.logits.detach().cpu().tolist() == pytest.approx([0.05, -0.05])


def test_completion_sum_excludes_prompt_and_does_not_normalize_length():
    model = TwoTokenModel()
    one = sequence_log_probability(model, FangRollout("a", (1, 1, 1), (0,)))
    two = sequence_log_probability(model, FangRollout("aa", (1,), (0, 0)))
    assert two.item() == pytest.approx(2 * one.item())
    with pytest.raises(ValueError, match="token IDs"):
        sequence_log_probability(model, FangRollout("a"))


def test_batch_mean_includes_constant_reward_groups():
    model = TwoTokenModel()
    policy = HuggingFaceFangPolicy(model, None, learning_rate=0.1)
    responses = [FangRollout("a", (0,), (0,)), FangRollout("b", (0,), (1,))]
    informative = FangGroup(FangExample("q", "a"), responses, [1.0, 0.0], [True, False], None)
    constant = FangGroup(FangExample("r", "a"), responses, [1.0, 1.0], [True, True], None)
    policy.update([informative, constant])
    assert model.logits.detach().tolist() == pytest.approx([0.025, -0.025])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_huggingface_rollout_checkpoint_roundtrip(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable in this process")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    backend = Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "42": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
    )
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=4, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
        bos_token_id=1, eos_token_id=1, pad_token_id=0,
    ))
    model.save_pretrained(tmp_path / "base")
    tokenizer.save_pretrained(tmp_path / "base")
    policy = HuggingFaceFangPolicy.from_pretrained({
        "model_name": str(tmp_path / "base"), "learning_rate": 0.1,
        "max_prompt_tokens": 128, "max_new_tokens": 3,
        "lora_rank": 2, "lora_alpha": 4, "lora_target_modules": ["q_proj", "v_proj"],
    }, device)
    samples = policy.sample("42", 2)
    assert len(samples) == 2
    assert all(1 <= len(sample.completion_ids) <= 3 for sample in samples)
    assert all(sample.prompt_ids for sample in samples)
    before_parameters = {name: parameter.detach().clone() for name, parameter in policy.model.named_parameters()}
    policy.update([FangGroup(
        FangExample("q", "a"),
        [FangRollout("a", (2, 3), (3, 1)), FangRollout("b", (2, 3), (2, 1))],
        [1.0, 0.0], [True, False], None,
    )])
    changed = [name for name, parameter in policy.model.named_parameters()
               if not torch.equal(parameter, before_parameters[name])]
    assert changed
    assert all("lora_" in name for name in changed)
    before = policy.answer("42")[0]
    policy.save(tmp_path / "checkpoint")
    restored = HuggingFaceFangPolicy.from_checkpoint(tmp_path / "checkpoint", device)
    assert restored.answer("42")[0] == before
    assert all(not parameter.requires_grad for parameter in restored.model.parameters())
