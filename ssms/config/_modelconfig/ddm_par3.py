"""Configuration for DDM par3 models."""

import cssm
from ssms.basic_simulators import boundary_functions as bf


def get_ddm_par3_config():
    """Get the configuration for the DDM par3 model.

    ``ddm_par3`` is ``ddm_par2`` plus a signed, time-accumulating coherence
    drive.  The high accumulator's instantaneous drift is
    ``vh + gain * coh * t_elapsed``; the low accumulator picked out by
    ``sign(coh)`` (walker 1 for ``coh < 0``, walker 2 for ``coh > 0``) gets
    ``vl + gain * |coh| * t_elapsed``.  ``gain = 0`` recovers ``ddm_par2``
    exactly; negative ``gain`` makes the drive decay with time.
    """
    return {
        "name": "ddm_par3",
        "params": ["vh", "vl1", "vl2", "coh", "gain", "a", "zh", "zl1", "zl2", "t"],
        "param_bounds": [
            [-4.0, -4.0, -4.0, -1.0, -2.0, 0.3, 0.2, 0.2, 0.2, 0.0],
            [4.0, 4.0, 4.0, 1.0, 2.0, 2.5, 0.8, 0.8, 0.8, 2.0],
        ],
        "boundary_name": "constant",
        "boundary": bf.constant,
        "n_params": 10,
        "default_params": [0.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.5, 0.5, 0.5, 1.0],
        "nchoices": 4,
        "choices": [0, 1, 2, 3],
        "n_particles": 1,
        "simulator": cssm.ddm_flexbound_par3,
        "parameter_transforms": {
            "sampling": [],
            "simulation": [],
        },
    }
