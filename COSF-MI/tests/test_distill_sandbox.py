"""Tests for the execution sandbox and parsers in scripts/generate_cosf_distill.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import generate_cosf_distill as gen  # noqa: E402

GOOD = "x = 5 + 5\nassert x == 10\n"

TEACHER_NUMERIC = """QUESTION: What is 5+5?
ANSWER: 5 plus 5 is **10**.
```python
answer = 5 + 5
assert answer == 10
```"""


def test_passing_code_is_kept():
    r = gen.run_sandboxed(GOOD)
    assert r.ok and r.reason == "ok"


def test_math_import_allowed():
    r = gen.run_sandboxed("import math\nassert abs(math.sin(math.pi)) < 1e-9\n")
    assert r.ok, r.detail


def test_syntax_error_rejected():
    r = gen.run_sandboxed("x = (5 +\nassert x == 10\n")
    assert not r.ok and r.reason == "syntax"


def test_failed_assert_rejected():
    r = gen.run_sandboxed("assert 5 + 5 == 11\n")
    assert not r.ok and r.reason == "error" and "AssertionError" in r.detail


def test_exception_rejected():
    r = gen.run_sandboxed("x = 1 / 0\nassert x\n")
    assert not r.ok and r.reason == "error" and "ZeroDivisionError" in r.detail


def test_infinite_loop_killed_by_timeout():
    r = gen.run_sandboxed("while True:\n    pass\nassert True\n", timeout=1.0)
    assert not r.ok and r.reason == "timeout"
    assert r.seconds < 3.0


def test_memory_hog_killed():
    code = "chunks = []\nwhile True:\n    chunks.append(bytearray(10_000_000))\nassert True\n"
    r = gen.run_sandboxed(code, timeout=5.0, mem_limit_mb=200)
    assert not r.ok and r.reason == "memory"


def test_chatty_script_does_not_hang():
    r = gen.run_sandboxed("for i in range(200000):\n    print('x' * 50)\nassert True\n")
    assert r.ok, r.detail


def test_unsafe_code_never_runs():
    for code in ["import os\nassert True\n", "import subprocess\nassert True\n",
                 "open('x.txt', 'w')\nassert True\n", "__import__('os')\nassert True\n",
                 "().__class__.__bases__\nassert True\n"]:
        r = gen.run_sandboxed(code)
        assert not r.ok and r.reason == "unsafe", code


def test_code_without_assert_rejected():
    r = gen.run_sandboxed("x = 5 + 5\nprint(x)\n")
    assert not r.ok and r.reason == "no_assert"


def test_extract_code_blocks():
    assert gen.extract_code_blocks(TEACHER_NUMERIC) == ["answer = 5 + 5\nassert answer == 10"]


def test_numeric_sample_end_to_end():
    task = {"category": "numeric", "topic": "arithmetic"}
    record, reason, _ = gen.verify(task, TEACHER_NUMERIC, timeout=5.0, mem_limit_mb=512)
    assert reason == "ok"
    assert record["line"] == "[INST] What is 5+5? [/INST] 5 plus 5 is 10."


def test_wrong_teacher_answer_dropped():
    wrong = TEACHER_NUMERIC.replace("assert answer == 10", "assert answer == 11")
    record, reason, _ = gen.verify({"category": "numeric", "topic": "arithmetic"}, wrong, 5.0, 512)
    assert record is None and reason == "error"


def test_dialogue_filler_dropped():
    text = "USER: Is the GPU free?\nAURA: Great question! Yes, it is idle."
    record, reason, _ = gen.verify({"category": "dialogue", "topic": "gpu"}, text, 5.0, 512)
    assert record is None and reason == "filler"


def test_dialogue_kept_and_formatted_for_finetune_chat():
    text = "USER: Is the GPU free?\nAURA: Yes, it is idle at 2%.\nUSER: Start training?\nAURA: Sure, starting it now."
    record, reason, _ = gen.verify({"category": "dialogue", "topic": "gpu"}, text, 5.0, 512)
    assert reason == "ok"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from finetune_chat import parse_conversation
    assert parse_conversation(record["line"]) == record["turns"]


def test_too_many_tokens_dropped():
    count = gen.make_token_counter()
    long_reply = "The GPU is idle and ready. " * 6
    text = "\n".join(f"USER: Is the GPU free, turn {i}?\nAURA: {long_reply}" for i in range(4))
    record, reason, detail = gen.verify({"category": "dialogue", "topic": "gpu"}, text, 5.0, 512, count, 240)
    assert record is None and reason == "too_many_tokens", detail
    ok, reason, _ = gen.verify({"category": "numeric", "topic": "arithmetic"}, TEACHER_NUMERIC, 5.0, 512, count, 240)
    assert reason == "ok" and count(ok["turns"]) < 30


def test_code_first_format_parses():
    text = "QUESTION: What is 2+3?\n```python\nr = 2 + 3\nassert r == 5\n```\nANSWER: 2 plus 3 is 5."
    assert gen.parse_numeric(text) == ("What is 2+3?", "2 plus 3 is 5.", "r = 2 + 3\nassert r == 5")


def test_dedup_key_ignores_case_and_punctuation():
    assert gen.norm_key([("What is 5+5?", "10")]) == gen.norm_key([("what is 55", "ten")])


COMPUTED = {"category": "numeric", "topic": "complex", "computed": True}


def test_computed_answer_filled_from_code():
    text = ("QUESTION: If z = 3+4i and c = -2-i, what is |z*z + c|?\n```python\n"
            "z = complex(3, 4)\nc = complex(-2, -1)\nresult = abs(z * z + c)\n```\n"
            "ANSWER: The magnitude is about RESULT.")
    record, reason, detail = gen.verify(COMPUTED, text, 5.0, 512)
    assert reason == "ok", detail
    assert record["turns"][0][1] == "The magnitude is about 24.7."  # the teacher had guessed 5.1


def test_computed_answer_rejects_complex_or_missing_placeholder():
    bad_value = "QUESTION: q?\n```python\nresult = complex(1, 2)\n```\nANSWER: It is RESULT."
    assert gen.verify(COMPUTED, bad_value, 5.0, 512)[1] == "bad_result"
    no_slot = "QUESTION: q?\n```python\nresult = 2\n```\nANSWER: It is 2."
    assert gen.verify(COMPUTED, no_slot, 5.0, 512)[1] == "format"


def test_format_result():
    assert [gen.format_result(v) for v in (3.0, 2.5, 0.8660254, 7)] == ["3", "2.5", "0.87", "7"]
