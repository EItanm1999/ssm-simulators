"""Tests for the Chong-class model additions.

Covers three models added on top of the existing par2/seq2 family:

* ``ddm_par2_sv_no_bias`` / ``ddm_seq2_sv_no_bias`` -- between-sample drift
  variability (``svh`` on the high accumulator, ``svl`` on both low ones).
* ``ddm_par3`` -- par2 plus a signed, time-accumulating coherence drive scaled
  by ``gain``.

Each new model is checked for basic sanity (legal choice codes, RTs above the
non-decision time, no NaNs) and for exact reduction to its parent model at the
parameter values where the extension switches off (``sv = 0``, ``gain = 0``).
"""

import numpy as np
import pytest

from ssms.basic_simulators.simulator import OMISSION_SENTINEL, simulator
from ssms.config import model_config

N_SAMPLES = 5000
SEED = 20260915

# Tolerances for the "reduces to the parent model" comparisons. The child model
# consumes a different number of random draws per sample than its parent, so the
# two streams diverge; only the sampling distributions can be compared.
CHOICE_P_TOL = 0.03
MEDIAN_RT_TOL = 0.05

NEW_MODELS = ["ddm_par2_sv_no_bias", "ddm_seq2_sv_no_bias", "ddm_par3"]


def _default_theta(model: str) -> dict:
    """Parameter dict built from the model config's own defaults."""
    cfg = model_config[model]
    return dict(zip(cfg["params"], cfg["default_params"]))


def _simulate(model: str, theta: dict, n_samples: int = N_SAMPLES, seed: int = SEED):
    out = simulator(theta=theta, model=model, n_samples=n_samples, random_state=seed)
    return np.asarray(out["rts"]).flatten(), np.asarray(out["choices"]).flatten()


def _choice_props(choices: np.ndarray) -> np.ndarray:
    return np.array([(choices == c).mean() for c in (0, 1, 2, 3)])


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", NEW_MODELS)
def test_model_is_registered(model):
    cfg = model_config[model]
    assert cfg["n_params"] == len(cfg["params"]) == len(cfg["default_params"])
    assert len(cfg["param_bounds"][0]) == len(cfg["param_bounds"][1]) == cfg["n_params"]
    # The key main's config builder validates.
    assert set(cfg["parameter_transforms"]) == {"sampling", "simulation"}


@pytest.mark.parametrize("model", NEW_MODELS)
def test_defaults_simulate_cleanly(model):
    theta = _default_theta(model)
    rts, choices = _simulate(model, theta)

    assert rts.shape == (N_SAMPLES,)
    assert choices.shape == (N_SAMPLES,)
    assert set(np.unique(choices)).issubset({0, 1, 2, 3})
    assert not np.isnan(rts).any()
    assert not np.isnan(choices.astype(float)).any()
    assert not (rts == OMISSION_SENTINEL).any()
    # Every RT must exceed the non-decision time, hence be strictly positive.
    assert (rts > theta["t"]).all()


@pytest.mark.parametrize("model", NEW_MODELS)
def test_all_four_choices_are_reachable(model):
    """Choice codes are 2 * [high hit upper] + [low hit upper]; 4 is illegal."""
    theta = _default_theta(model)
    _, choices = _simulate(model, theta)
    assert set(np.unique(choices)) == {0, 1, 2, 3}
    assert (choices != 4).all()


# ---------------------------------------------------------------------------
# Reduction to the parent models
# ---------------------------------------------------------------------------


PAR2_BASE = {"vh": 0.8, "vl1": 0.5, "vl2": -0.4, "a": 1.2, "t": 0.3}


@pytest.mark.parametrize(
    "child, parent",
    [
        ("ddm_par2_sv_no_bias", "ddm_par2_no_bias"),
        ("ddm_seq2_sv_no_bias", "ddm_seq2_no_bias"),
    ],
)
def test_sv_zero_reduces_to_parent(child, parent):
    parent_rts, parent_choices = _simulate(parent, dict(PAR2_BASE))
    child_rts, child_choices = _simulate(child, dict(PAR2_BASE, svh=0.0, svl=0.0))

    np.testing.assert_allclose(
        _choice_props(child_choices),
        _choice_props(parent_choices),
        atol=CHOICE_P_TOL,
    )
    assert abs(np.median(child_rts) - np.median(parent_rts)) < MEDIAN_RT_TOL


def test_par3_gain_zero_reduces_to_par2():
    parent_theta = dict(PAR2_BASE, zh=0.5, zl1=0.5, zl2=0.5)
    child_theta = dict(parent_theta, coh=0.0, gain=0.0)

    parent_rts, parent_choices = _simulate("ddm_par2", parent_theta)
    child_rts, child_choices = _simulate("ddm_par3", child_theta)

    np.testing.assert_allclose(
        _choice_props(child_choices),
        _choice_props(parent_choices),
        atol=CHOICE_P_TOL,
    )
    assert abs(np.median(child_rts) - np.median(parent_rts)) < MEDIAN_RT_TOL


def test_par3_gain_zero_reduces_to_par2_at_nonzero_coh():
    """gain = 0 kills the drive whatever the coherence is."""
    parent_theta = dict(PAR2_BASE, zh=0.5, zl1=0.5, zl2=0.5)
    child_theta = dict(parent_theta, coh=0.9, gain=0.0)

    parent_rts, parent_choices = _simulate("ddm_par2", parent_theta)
    child_rts, child_choices = _simulate("ddm_par3", child_theta)

    np.testing.assert_allclose(
        _choice_props(child_choices),
        _choice_props(parent_choices),
        atol=CHOICE_P_TOL,
    )
    assert abs(np.median(child_rts) - np.median(parent_rts)) < MEDIAN_RT_TOL


# ---------------------------------------------------------------------------
# The extensions actually do something
# ---------------------------------------------------------------------------


# Drift variability widens the RT distribution only where the mean drift is
# already strong enough that the base RT distribution is fast and tight. At a
# weak mean drift (PAR2_BASE) raising svh mostly *raises* E|v| and therefore
# narrows the bulk of the RT distribution, so the widening claim is checked in
# the strong-drift regime.
PAR2_STRONG_DRIFT = {"vh": 3.0, "vl1": 3.0, "vl2": 3.0, "a": 1.5, "t": 0.3}


@pytest.mark.parametrize("seed", [SEED, 1, 2])
def test_par2_sv_increasing_svh_widens_rt_iqr(seed):
    def iqr(svh):
        rts, _ = _simulate(
            "ddm_par2_sv_no_bias",
            dict(PAR2_STRONG_DRIFT, svh=svh, svl=0.0),
            seed=seed,
        )
        return np.subtract(*np.percentile(rts, [75, 25]))

    assert iqr(1.5) > iqr(0.0)


@pytest.mark.parametrize("model", ["ddm_par2_sv_no_bias", "ddm_seq2_sv_no_bias"])
def test_sv_changes_the_rt_distribution(model):
    """Turning sv on must move the RT distribution at all."""
    rts_off, _ = _simulate(model, dict(PAR2_STRONG_DRIFT, svh=0.0, svl=0.0))
    rts_on, _ = _simulate(model, dict(PAR2_STRONG_DRIFT, svh=1.5, svl=1.5))
    assert rts_on.std() > rts_off.std()


def test_par3_positive_gain_speeds_the_high_accumulator():
    """A positive gain with positive coh drives the high walker upward."""
    base = dict(PAR2_BASE, zh=0.5, zl1=0.5, zl2=0.5, coh=1.0)
    _, choices_off = _simulate("ddm_par3", dict(base, gain=0.0))
    _, choices_on = _simulate("ddm_par3", dict(base, gain=2.0))

    # choices 2 and 3 are the "high accumulator hit the upper bound" outcomes.
    p_high_off = np.isin(choices_off, [2, 3]).mean()
    p_high_on = np.isin(choices_on, [2, 3]).mean()
    assert p_high_on > p_high_off


def test_par3_negative_gain_reverses_the_drive():
    base = dict(PAR2_BASE, zh=0.5, zl1=0.5, zl2=0.5, coh=1.0)
    _, choices_pos = _simulate("ddm_par3", dict(base, gain=2.0))
    _, choices_neg = _simulate("ddm_par3", dict(base, gain=-2.0))

    p_high_pos = np.isin(choices_pos, [2, 3]).mean()
    p_high_neg = np.isin(choices_neg, [2, 3]).mean()
    assert p_high_neg < p_high_pos
