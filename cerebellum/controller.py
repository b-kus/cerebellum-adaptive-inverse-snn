from brian2 import second
from config.constants import (
    KP, KD, KI_PD, DECODE_GAIN, CONTROL_DT, TORQUE_LIMIT
)
from cerebellum.snn_model import CerebellarSNN
from cerebellum.pendulum_env import pid_with_integral
import numpy as np


class PDController:
    """Baseline PD/PID controller (no SNN)."""
    def __init__(self, Kp=KP, Kd=KD, Ki=KI_PD):
        self.Kp, self.Kd, self.Ki = Kp, Kd, Ki
        self.integral = 0.0

    def __call__(self, angle, omega, target, Kp=None, Kd=None, Ki=None, integral=None):
        torque, self.integral = pid_with_integral(
            angle, omega, target, self.Kp, self.Kd, self.Ki, self.integral
        )
        return torque, self.integral

    def reset(self):
        self.integral = 0.0

class OnlineSNNPIDController:
    """PID controller combined with an online-learning SNN (Kawato FEL)."""
    def __init__(self, snn: CerebellarSNN, Kp=KP, Kd=KD, Ki=KI_PD,
                 learning_enabled=True,
                 smoothing_alpha=0.05,
                 decode_gain=DECODE_GAIN):
        self.snn = snn
        self.Kp, self.Kd, self.Ki = Kp, Kd, Ki
        self.learning_enabled = learning_enabled
        self.smoothing_alpha = smoothing_alpha
        self.decode_gain = decode_gain
        self._smoothed_snn_torque = 0.0
        self.integral = 0.0

    def reset(self):
        self._smoothed_snn_torque = 0.0
        self.integral = 0.0
        self.snn.reset_dynamic_state()

    def __call__(self, angle, omega, target, Kp=None, Kd=None, Ki=None, integral=None):
        # 1. PID error (climbing fibre teaching signal)
        pid_torque, self.integral = pid_with_integral(
            angle, omega, target, self.Kp, self.Kd, self.Ki, self.integral
        )

        # 2. Encode state and set climbing fibre rates
        self.snn.encode(theta=angle, omega=omega, theta_d=target)
        if self.learning_enabled:
            self.snn.set_climbing_fibre_rates(pid_torque)
        else:
            self.snn.set_climbing_fibre_rates(0.0)

        # 3. SNN step
        raw_snn_torque = self.snn.step(window=CONTROL_DT * second,
                                        gain=self.decode_gain)
        a = self.smoothing_alpha
        self._smoothed_snn_torque = a * raw_snn_torque + (1 - a) * self._smoothed_snn_torque

        # 4. Combine torques
        total_torque = np.clip(pid_torque + self._smoothed_snn_torque,
                                -TORQUE_LIMIT, TORQUE_LIMIT)
        return total_torque, self.integral