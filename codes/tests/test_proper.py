"""Contracts for public algorithm taxonomy and experiment configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CODES_ROOT = Path(__file__).resolve().parents[1]
if str(_CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODES_ROOT))

from problems.init_policy import init_policy
from utils.alg.alg_bank import get_alg_bank


def test_primary_comparison_has_expected_algorithms() -> None:
    assert [name for name, _ in get_alg_bank("MainComp")] == [
        "DisGrem",
        "CeDisGrem",
        "AdaDisGrem",
        "CeAdaDisGrem",
        "EXTRA",
        "DIGing",
        "DQM",
        "ESOM",
        "SONATA",
        "NetworkGIANT",
    ]


def test_unknown_algorithm_group_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown test_key"):
        get_alg_bank("TypoedGroup")


def test_paper_policy_covers_all_nine_objectives() -> None:
    params, policy, starts = init_policy("regular")
    expected = {
        "ridge",
        "quadbad",
        "huber",
        "logsumexp",
        "linlog",
        "rosenbrock",
        "styblinski_tang",
        "logreg_real",
        "logreg_ncvr",
    }

    assert set(params) == expected
    assert set(policy) == expected
    assert set(starts) == expected
    for objective in expected:
        assert policy[objective]["M_factor"] > 0
        assert policy[objective]["alpha"] > 0
        assert policy[objective]["maxIt"] > 0
