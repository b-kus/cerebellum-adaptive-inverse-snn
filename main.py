# main.py
import os
import numpy as np
import matplotlib.pyplot as plt
from config.constants import *
from cerebellum import CerebellarSNN, PDController, OnlineSNNPIDController, run_pendulum
from utils import compute_metrics

def main():
    print("=" * 70)
    print("Cerebellar SNN Feedback-Error-Learning")
    print(f"Target = {TARGET_ANGLE} rad, Ki_PD = {KI_PD}, Ki_PID = {KI_PID}")
    print(f"CF_SCALE = {CF_SCALE}, ETA_LTD = {ETA_LTD}, LTP_RATE = {LTP_RATE}")
    print("=" * 70)

    # EXPERIMENT 1: PD vs PD + SNN 
    print("\n--- Experiment 1: PD vs PD+SNN (KI=0) ---")

    # PD only
    pd_controller = PDController(Kp=KP, Kd=KD, Ki=KI_PD)
    pd_controller.reset()
    print("Running PD only...")
    controls_pd, time_pd, angles_pd, targets_pd = run_pendulum(
        controller=pd_controller,
        duration=TRAIN_DURATION,
        target_angle=TARGET_ANGLE,
        initial_angle=0.3,
        Ki=KI_PD,
        use_perturbation=True,
        perturb_at=PERTURB_AT,
        perturb_target=PERTURB_TARGET,
        verbose=True
    )

    # PD + SNN
    snn_pd = CerebellarSNN(rng_seed=42)
    snn_pd_controller = OnlineSNNPIDController(
        snn_pd, Kp=KP, Kd=KD, Ki=KI_PD, learning_enabled=True
    )
    snn_pd_controller.reset()
    print("Running PD + SNN (online learning)...")
    controls_pdsnn, time_pdsnn, angles_pdsnn, targets_pdsnn = run_pendulum(
        controller=snn_pd_controller,
        duration=TRAIN_DURATION,
        target_angle=TARGET_ANGLE,
        initial_angle=0.3,
        Ki=KI_PD,
        use_perturbation=True,
        perturb_at=PERTURB_AT,
        perturb_target=PERTURB_TARGET,
        verbose=True
    )

    rms_pd, _ = compute_metrics(angles_pd, targets_pd, controls_pd)
    rms_pdsnn, _ = compute_metrics(angles_pdsnn, targets_pdsnn, controls_pdsnn)
    print(f"PD only     : RMS error = {rms_pd:.4f} rad")
    print(f"PD + SNN    : RMS error = {rms_pdsnn:.4f} rad")

    # EXPERIMENT 2: PID vs PID + SNN 
    print("\n--- Experiment 2: PID vs PID+SNN (KI={:.1f}) ---".format(KI_PID))

    # PID only
    pid_controller = PDController(Kp=KP, Kd=KD, Ki=KI_PID)
    pid_controller.reset()
    print("Running PID only...")
    controls_pid, time_pid, angles_pid, targets_pid = run_pendulum(
        controller=pid_controller,
        duration=TRAIN_DURATION,
        target_angle=TARGET_ANGLE,
        initial_angle=0.3,
        Ki=KI_PID,
        use_perturbation=True,
        perturb_at=PERTURB_AT,
        perturb_target=PERTURB_TARGET,
        verbose=True
    )

    # PID + SNN
    snn_pid = CerebellarSNN(rng_seed=42)
    snn_pid_controller = OnlineSNNPIDController(
        snn_pid, Kp=KP, Kd=KD, Ki=KI_PID, learning_enabled=True
    )
    snn_pid_controller.reset()
    print("Running PID + SNN (online learning)...")
    controls_pidsnn, time_pidsnn, angles_pidsnn, targets_pidsnn = run_pendulum(
        controller=snn_pid_controller,
        duration=TRAIN_DURATION,
        target_angle=TARGET_ANGLE,
        initial_angle=0.3,
        Ki=KI_PID,
        use_perturbation=True,
        perturb_at=PERTURB_AT,
        perturb_target=PERTURB_TARGET,
        verbose=True
    )

    rms_pid, _ = compute_metrics(angles_pid, targets_pid, controls_pid)
    rms_pidsnn, _ = compute_metrics(angles_pidsnn, targets_pidsnn, controls_pidsnn)
    print(f"PID only    : RMS error = {rms_pid:.4f} rad")
    print(f"PID + SNN   : RMS error = {rms_pidsnn:.4f} rad")

    # PLOTS 
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex='col')

    axes[0, 0].plot(time_pd, angles_pd, color='red', alpha=0.7, label='PD only')
    axes[0, 0].plot(time_pdsnn, angles_pdsnn, color='blue', alpha=0.7, label='PD+SNN')
    axes[0, 0].plot(time_pdsnn, targets_pdsnn, color='k', linestyle='--', label='Target')
    axes[0, 0].set_title('Experiment 1: PD vs PD+SNN (Ki=0)')
    axes[0, 0].set_ylabel('Angle (rad)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(time_pid, angles_pid, color='red', alpha=0.7, label='PID only')
    axes[0, 1].plot(time_pidsnn, angles_pidsnn, color='blue', alpha=0.7, label='PID+SNN')
    axes[0, 1].plot(time_pidsnn, targets_pidsnn, color='k', linestyle='--', label='Target')
    axes[0, 1].set_title('Experiment 2: PID vs PID+SNN (Ki=2.0)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(time_pd, np.abs(targets_pd - angles_pd), color='red', alpha=0.7, label='|error| PD only')
    axes[1, 0].plot(time_pdsnn, np.abs(targets_pdsnn - angles_pdsnn), color='blue', alpha=0.7, label='|error| PD+SNN')
    axes[1, 0].set_ylabel('|Angle error| (rad)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(time_pid, np.abs(targets_pid - angles_pid), color='red', alpha=0.7, label='|error| PID only')
    axes[1, 1].plot(time_pidsnn, np.abs(targets_pidsnn - angles_pidsnn), color='blue', alpha=0.7, label='|error| PID+SNN')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()

    print("\nDone.")

if __name__ == '__main__':
    main()