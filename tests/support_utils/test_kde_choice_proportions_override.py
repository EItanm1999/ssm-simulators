"""Opt-in override of KDE choice proportions.

Callers that pad or oversample the per-choice RT arrays (e.g. stratified
resampling to give a rare choice a larger KDE support) can pass the TRUE
per-choice proportions via ``simulator_data['choice_proportions_override']``.
Absent the key, behavior is exactly as before.
"""

import numpy as np

from ssms.support_utils.kde_class import LogKDE


def _simulator_data(rts, choices, override=None):
    d = {
        "rts": np.asarray(rts, dtype=float).reshape(-1, 1),
        "choices": np.asarray(choices, dtype=float).reshape(-1, 1),
        "metadata": {"max_t": 20.0, "possible_choices": [-1, 1]},
    }
    if override is not None:
        d["choice_proportions_override"] = override
    return d


def _make(n=4000, frac_pos=0.5, override=None, seed=3):
    rng = np.random.default_rng(seed)
    rts = rng.lognormal(mean=-0.3, sigma=0.4, size=n)
    choices = np.where(rng.uniform(size=n) < frac_pos, 1.0, -1.0)
    return LogKDE(_simulator_data(rts, choices, override))


def test_default_proportions_from_raw_counts():
    kde = _make(frac_pos=0.8)
    props = dict(zip(kde.data["choices"], kde.data["choice_proportions"]))
    assert abs(props[1.0] - 0.8) < 0.03
    assert abs(sum(props.values()) - 1.0) < 1e-12


def test_override_bypasses_raw_counts():
    # data is padded 50/50, but the TRUE proportions are 90/10
    kde = _make(frac_pos=0.5, override={1.0: 0.9, -1.0: 0.1})
    props = dict(zip(kde.data["choices"], kde.data["choice_proportions"]))
    assert props[1.0] == 0.9
    assert props[-1.0] == 0.1
