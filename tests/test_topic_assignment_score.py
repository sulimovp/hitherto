"""Offline fixture for scripts/score_topic_assignment.py."""

from casefile.predict.assignment_score import (
    automatic_assign,
    score_assignment,
    wilson_interval,
)


def test_wilson_interval_contains_point():
    lo, hi = wilson_interval(86, 100)
    assert 0.0 <= lo < 0.86 < hi <= 1.0


def test_assigner_does_not_match_masked_select():
    assert automatic_assign("torch.masked_select OOM", "", "torch/masked") is False
    assert automatic_assign("torch.masked.mean is slow", "", "torch/masked") is True
    assert automatic_assign("please add MaskedTensor permute", "", "torch/masked") is True


def test_assigner_rejects_competing_title_and_boolean_masking():
    assert automatic_assign(
        "NestedTensor max_seqlen compile failure",
        "also mentions MaskedTensor in passing later " * 20,
        "torch/masked",
    ) is False
    assert automatic_assign(
        "NaN in MSELoss",
        "I get nan when using MSELoss on masked tensors via x[mask].",
        "torch/masked",
    ) is False


def test_assigner_accepts_ctor_in_lead():
    assert automatic_assign(
        "TypeError: mask must have dtype bool",
        "I use MaskedTensor(). I give it the mask:",
        "torch/masked",
    ) is True


def test_apertus_assigner_needs_anchor_not_generic_chat_template():
    syn = ("apertus-format", "chat template", "tool call parser")
    assert (
        automatic_assign(
            "[docs] chat template",
            "adds docs for loading modern chat template formats",
            "apertus format",
            synonyms=syn,
            repo="huggingface/transformers",
        )
        is False
    )
    assert (
        automatic_assign(
            "Apertus tool parser",
            "enable --tool-call-parser and --chat-template in vLLM",
            "apertus format",
            synonyms=syn,
            repo="swiss-ai/Apertus-8B-Instruct-2509",
        )
        is True
    )
    assert (
        automatic_assign(
            "EOS token",
            "the format should emit EOS",
            "apertus format",
            synonyms=syn,
            repo="swiss-ai/apertus-format",
        )
        is True
    )


def test_labelled_pytorch_holdout_is_below_perfect_precision():
    from pathlib import Path

    import yaml

    labelled = (
        Path(__file__).resolve().parents[1]
        / "eval"
        / "topic_assignment"
        / "pytorch_holdout.yaml"
    )
    data = yaml.safe_load(labelled.read_text(encoding="utf-8"))
    result = score_assignment(list(data["items"]), path="torch/masked")
    assert result["n"] >= 90
    assert result["fp"] >= 1
    assert result["precision"] < 1.0
    assert result["precision"] >= 0.80


def test_labelled_apertus_set_meets_precision_floor():
    from pathlib import Path

    import yaml

    labelled = (
        Path(__file__).resolve().parents[1] / "eval" / "topic_assignment" / "apertus.yaml"
    )
    data = yaml.safe_load(labelled.read_text(encoding="utf-8"))
    result = score_assignment(
        list(data["items"]),
        path=data["path"],
        synonyms=tuple(data["synonyms"]),
    )
    assert result["n"] >= 80
    assert result["precision"] >= 0.80
    assert result["fn"] >= 1
    assert result["precision"] < 1.0


def test_score_precision_recall_from_fixture():
    items = [
        {
            "repo": "pytorch/pytorch",
            "number": 1,
            "topic": "torch/masked",
            "title": "bug in torch/masked",
            "body": "",
        },
        {
            "repo": "pytorch/pytorch",
            "number": 2,
            "topic": "torch/masked",
            "title": "please help",
            "body": "no path",
        },
        {
            "repo": "pytorch/pytorch",
            "number": 3,
            "topic": None,
            "title": "nested tensor docs",
            "body": "",
        },
        {
            "repo": "pytorch/pytorch",
            "number": 4,
            "topic": None,
            "title": "torch/masked mention by mistake",
            "body": "",
        },
    ]
    result = score_assignment(items, path="torch/masked")
    assert result["tp"] == 1
    assert result["fn"] == 1
    assert result["fp"] == 1
    assert result["tn"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
