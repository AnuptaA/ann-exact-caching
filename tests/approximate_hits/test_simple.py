#!/usr/bin/env python3
"""Tests for the analytic scores in eval_simple."""


import numpy as np

from simulations.approximate_hits import eval_simple as es
from utils.cache.shell import COUNT_SCALES, SAMPLE_J


def _shell(counts):
    S = np.zeros((len(counts), SAMPLE_J + len(COUNT_SCALES)))
    S[:, SAMPLE_J] = np.log1p(counts)
    return S


def test_best_fit_l2_takes_min_over_balls():
    B = np.array([[0.6, 0.8, 3.0, 4.0]])          # balls with fits 1.0 and 5.0
    assert np.isclose(es.best_fit_l2(B)[0], 1.0)


def test_analytic_orders_by_count_then_fit():
    B = np.array([[0.3, 0.4, 9, 9], [3.0, 4.0, 9, 9], [0.3, 0.4, 9, 9]])  # fits .5, 5, .5
    s = es.analytic(B, _shell([0, 0, 1]))
    # zero seen intruders beat one; among zeros the better fit wins
    assert s[0] > s[1] > s[2]


def test_monotone_constraint_layout():
    cst = es.monotone_cst(6, SAMPLE_J + len(COUNT_SCALES))
    assert cst == [-1] * 6 + [1] * SAMPLE_J + [-1] * len(COUNT_SCALES)
