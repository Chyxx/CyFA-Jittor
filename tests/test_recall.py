import pytest

from eval.recall_eval import contains_score, format_document


@pytest.mark.parametrize(
    "task,doc,prompt,answers",
    [
        (
            "based_fda",
            {"text": "Aspirin is useful.\n", "key": "drug", "value": "aspirin"},
            "Aspirin is useful. Drug:",
            ["aspirin"],
        ),
        (
            "based_swde",
            {
                "text": "Some information. Summary of information above...tail",
                "key": "name",
                "value": "Alice",
            },
            "Some information. Name:",
            ["Alice"],
        ),
        (
            "based_squad",
            {"text": " passage Question? ", "value": "answer"},
            "passage Question?",
            ["answer"],
        ),
        (
            "based_nq_2048",
            {"context": "World. Who?", "question": "Who?", "answers": ["Alice"]},
            "World. Who?",
            ["Alice"],
        ),
        (
            "based_triviaqa",
            {"context": "[DOC]World. Who?", "question": "Who?", "answers": ["Alice"]},
            "World. Who?",
            ["Alice"],
        ),
        (
            "based_drop",
            {"context": "World. Who?", "question": "Who?", "answers": ["Alice"]},
            "World. Who?",
            ["Alice"],
        ),
    ],
)
def test_recall_prompts(task, doc, prompt, answers):
    assert format_document(task, doc) == (prompt, answers)
    assert contains_score("It is ALICE.", ["Alice"]) == 1
    assert contains_score("Nobody", ["Alice"]) == 0


def test_recall_context_selection_and_filtering():
    from cyclic_flow_attention.utils import Tokenizer
    from eval.recall_eval import prepare_document

    tokenizer = Tokenizer("byte")
    doc = {"text": "x" * 200 + "ANSWER" + "y" * 200, "value": "ANSWER"}
    ids, answers = prepare_document(
        "based_squad", doc, tokenizer, context_length=64, answer_length=16
    )
    assert len(ids) == 48
    assert tokenizer.decode(ids) == "x" * 24 + "ANSWER" + "y" * 18
    assert answers == ["ANSWER"]
    assert prepare_document("based_squad", {**doc, "value": "A"}, tokenizer) is None
