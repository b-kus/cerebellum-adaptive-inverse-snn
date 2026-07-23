import numpy as np

def compute_metrics(angles, targets, controls):
    """
    Compute RMS error and mean absolute torque.

    Args:
        angles: array of actual angles
        targets: array of target angles
        controls: array of torque commands

    Returns:
        rms_error: root-mean-square error (rad)
        mean_abs_torque: mean absolute torque (Nm)
    """
    angles = np.array(angles)
    targets = np.array(targets)
    controls = np.array(controls)
    error = targets - angles
    rms_error = np.sqrt(np.mean(error ** 2))
    mean_abs_torque = np.mean(np.abs(controls))
    return rms_error, mean_abs_torque

def target_angle_at(t, base_target, perturb_at=None, perturb_target=None):
    """
    Return the target angle at time t, with optional step change.

    Args:
        t: current time (s)
        base_target: base target angle (rad)
        perturb_at: time of perturbation (s), or None
        perturb_target: new target after perturbation (rad)

    Returns:
        target angle at time t (rad)
    """
    if perturb_at is not None and t >= perturb_at:
        return perturb_target
    return base_target