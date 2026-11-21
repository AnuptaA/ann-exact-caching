"""TriviaQA question encoding in resumable parts, with progress."""

import json
import logging

import numpy as np
import pytest

from workloads.q1.datasets import triviaqa

QUESTIONS = [f"question {i}" for i in range(23)]


def encoder(calls):
    def encode(texts):
        calls.append(len(texts))
        return np.array([[float(t.split()[1])] * triviaqa.DIM for t in texts])
    return encode


def test_parts_are_resumable_and_match_a_single_pass(tmp_path):
    calls = []
    inner = encoder(calls)

    def dying(texts):
        if len(calls) == 3:
            raise KeyboardInterrupt
        return inner(texts)

    with pytest.raises(KeyboardInterrupt):
        triviaqa.encode_questions(QUESTIONS, dying, batch_size=2, parts_dir=tmp_path / "p", part_size=6)
    assert sorted(p.name for p in (tmp_path / "p").glob("part_*")) == ["part_00000.npy"]
    resumed = []
    out = triviaqa.encode_questions(QUESTIONS, encoder(resumed), batch_size=2, parts_dir=tmp_path / "p",
                                    part_size=6)
    assert sum(resumed) == 17                      # the first part (6 questions) was not encoded again
    assert np.array_equal(out, triviaqa.encode_questions(QUESTIONS, encoder([]), batch_size=5))
    assert out.dtype == np.float32 and out[7, 0] == 7.0           # raw, in order


def test_parts_of_other_questions_are_refused(tmp_path):
    triviaqa.encode_questions(QUESTIONS, encoder([]), parts_dir=tmp_path / "p", part_size=10)
    with pytest.raises(ValueError, match="other questions"):
        triviaqa.encode_questions(QUESTIONS[:-1], encoder([]), parts_dir=tmp_path / "p", part_size=10)


def test_progress_lines_have_a_rate_and_an_eta(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="workloads.q1.progress"):
        triviaqa.encode_questions(QUESTIONS, encoder([]), batch_size=4, progress_interval=0.0)
    lines = [r.message for r in caplog.records if r.message.startswith("triviaqa encode:")]
    assert len(lines) == 6 and "questions/s" in lines[0] and "/23 questions" in lines[-1]


def test_a_wrong_output_shape_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        triviaqa.encode_questions(["q"], lambda b: np.ones((len(b), 3)), parts_dir=tmp_path / "p")
