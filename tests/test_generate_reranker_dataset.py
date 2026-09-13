import json
from pathlib import Path

from scripts.kleine_anfragen.generate_reranker_dataset import (
    clean_question,
    generate_examples,
    load_documents,
)


def test_clean_question_strips_single_number():
    assert clean_question("1. Wie viele Personen?") == "Wie viele Personen?"


def test_clean_question_strips_combined_numbers():
    assert clean_question("1. 2. Wie viele Personen?") == "Wie viele Personen?"


def test_clean_question_no_number():
    assert clean_question("Wie viele Personen?") == "Wie viele Personen?"


def test_generate_examples_query_has_no_leading_number():
    doc = {
        "pdf_file": "1.pdf",
        "vorgang_id": "1",
        "pairs": [{"question": "1. Wie viele?", "answer": "Viele."}],
    }
    examples = generate_examples([doc])
    assert examples[0]["query"] == "Wie viele?"


def make_doc(n_pairs: int, vorgang_id: str = "1") -> dict:
    return {
        "pdf_file": f"{vorgang_id}.pdf",
        "vorgang_id": vorgang_id,
        "pairs": [
            {"question": f"Question {i}", "answer": f"Answer {i}"}
            for i in range(n_pairs)
        ],
    }


def test_single_pair_doc_no_negative():
    examples = generate_examples([make_doc(1)])
    assert examples == [{"query": "Question 0", "passage": "Answer 0", "label": 1}]


def test_multi_pair_positive_and_negative():
    examples = generate_examples([make_doc(3)])
    positives = [e for e in examples if e["label"] == 1]
    negatives = [e for e in examples if e["label"] == 0]
    assert len(positives) == 3
    assert len(negatives) == 3


def test_negative_is_shifted_answer():
    doc = make_doc(3)
    examples = generate_examples([doc])
    # question 0 should get answer 1 as negative
    q0_negative = next(
        e for e in examples if e["query"] == "Question 0" and e["label"] == 0
    )
    assert q0_negative["passage"] == "Answer 1"


def test_negative_wraps_around():
    doc = make_doc(3)
    examples = generate_examples([doc])
    # last question should get answer 0 as negative (round-robin wrap)
    q2_negative = next(
        e for e in examples if e["query"] == "Question 2" and e["label"] == 0
    )
    assert q2_negative["passage"] == "Answer 0"


def test_empty_doc_skipped():
    examples = generate_examples([{"pairs": []}])
    assert examples == []


def test_load_documents(tmp_path: Path):
    data = [make_doc(2, "10"), make_doc(1, "11")]
    file = tmp_path / "test.jsonl"
    file.write_text("\n".join(json.dumps(d) for d in data))
    docs = load_documents(file)
    assert len(docs) == 2


def test_load_documents_skips_bad_lines(tmp_path: Path, capsys):
    file = tmp_path / "bad.jsonl"
    file.write_text(json.dumps(make_doc(1)) + "\nnot json\n" + json.dumps(make_doc(1)))
    docs = load_documents(file)
    assert len(docs) == 2
    captured = capsys.readouterr()
    assert "skipping line 2" in captured.err
