"""Configuration for DDM par2 models with between-trial drift variability."""

import cssm
from ssms.basic_simulators import boundary_functions as bf
from ssms.transforms import LambdaAdaptation

import numpy as np


def get_ddm_par2_sv_no_bias_config():
    """Get the configuration for the DDM par2 sv (no bias) model.

    Parallel two-stage DDM in which each simulated sample draws its own drift
    rates: ``vh_sample ~ Normal(vh, svh)``, ``vl1_sample ~ Normal(vl1, svl)``
    and ``vl2_sample ~ Normal(vl2, svl)``.  Starting points are fixed at 0.5
    like the other ``no_bias`` variants.
    """
    return {
        "name": "ddm_par2_sv_no_bias",
        "params": ["vh", "vl1", "vl2", "svh", "svl", "a", "t"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, 0.0, 0.0, 0.3, 0.0],
            [4.0, 4.0, 4.0, 2.5, 2.5, 2.5, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 7,
        "default_params": [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_par2_sv,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [
                LambdaAdaptation(
                    lambda theta, cfg, n: (
                        theta.update(
                            {
                                "zh": np.tile(np.array([0.5], dtype=np.float32), n),
                                "zl1": np.tile(np.array([0.5], dtype=np.float32), n),
                                "zl2": np.tile(np.array([0.5], dtype=np.float32), n),
                            }
                        )
                        or theta
                    ),
                    name="add_z_defaults",
                )
            ],
        },
    }
