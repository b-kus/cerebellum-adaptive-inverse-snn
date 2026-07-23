import numpy as np
import mujoco
from tqdm import tqdm
from config.constants import (
    PHYS_DT, CONTROL_DT, CONTROL_EVERY, DELAY_STEPS,
    TORQUE_LIMIT, KP, KD, KI_PD
)
from utils.metrics import target_angle_at

PENDULUM_XML = """
<mujoco model="pendulum">
    <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
    <option gravity="0 0 -9.81" timestep="0.005" integrator="RK4"/>
    <visual>
        <!-- Set default offscreen resolution -->
        <global offwidth="1440" offheight="768" azimuth="90" elevation="-45"/>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
          markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
        <material name="base_mat" rgba="0.2 0.2 0.2 1"/>
        <material name="link_mat" rgba="0.8 0.3 0.3 1"/>
    </asset>

    <statistic center="0 0 2"/>


    <worldbody>
        <light pos="0 0 5" dir="0 0 -1" diffuse="1 1 1"/>
        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" pos="0 0 0"/>

        <body name="center" pos="0 0 2">
            <joint name="center_joint" type="hinge" axis="0 1 0" damping="0.01"/>
            <geom name="spoke1" type="capsule" fromto="0 0 0  0 0 1" size="0.05" material="link_mat" mass="1.0"/>  
            <site name="tip" type="sphere" pos="0 0 1" size="0.05" material="link_mat"/>
        </body>
    </worldbody>
    <actuator>
        <motor name="shoulder" joint="center_joint"/>
    </actuator>
</mujoco>"""

def pid_with_integral(angle, omega, target, Kp, Kd, Ki, integral,
                       dt=CONTROL_DT, integral_limit=5.0):
    error = target - angle
    integral = np.clip(integral + error * dt, -integral_limit, integral_limit)
    derivative = -omega
    torque = Kp * error + Kd * derivative + Ki * integral
    torque = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)
    return torque, integral

def run_pendulum(controller, duration=20.0, target_angle=0.0,
                  initial_angle=0.3, Kp=KP, Kd=KD, Ki=KI_PD,
                  delay_steps=DELAY_STEPS, control_every=CONTROL_EVERY,
                  use_perturbation=False, perturb_at=None, perturb_target=None,
                  verbose=True):
    """
    Run the MuJoCo pendulum simulation with a given controller.

    Args:
        controller: callable(angle, omega, target, Kp, Kd, Ki, integral) -> (torque, integral)
        duration: simulation duration (s)
        target_angle: base target angle (rad)
        initial_angle: initial pendulum angle (rad)
        Kp, Kd, Ki: PID gains
        delay_steps: transport delay in physics steps
        control_every: steps between controller updates
        use_perturbation: if True, change target mid-run
        perturb_at: time of perturbation (s)
        perturb_target: new target after perturbation (rad)
        verbose: print progress

    Returns:
        controls: array of torque commands
        time_steps: array of time stamps
        angles: array of pendulum angles
        targets: array of target angles
    """
    model = mujoco.MjModel.from_xml_string(PENDULUM_XML)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    if dt <= 0:
        dt = PHYS_DT
        model.opt.timestep = dt

    data.qpos[0] = initial_angle
    data.qvel[0] = 0.0
    mujoco.mj_forward(model, data)

    controls, time_steps, angles, targets = [], [], [], []
    torque_history = np.zeros(delay_steps + 1)
    current_torque = 0.0
    integral = 0.0

    steps = int(duration / dt)
    if verbose:
        print(f"Simulating {duration} s ({steps} steps)...")
        progress = tqdm(total=steps, desc="Progress", unit="step")

    for i in range(steps):
        t = i * dt
        angle = data.qpos[0]
        omega = data.qvel[0]

        if use_perturbation:
            current_target = target_angle_at(t, target_angle, perturb_at, perturb_target)
        else:
            current_target = target_angle

        if i % control_every == 0:
            current_torque, integral = controller(
                angle, omega, current_target, Kp, Kd, Ki, integral
            )
            current_torque = np.clip(current_torque, -TORQUE_LIMIT, TORQUE_LIMIT)

        torque_history = np.roll(torque_history, -1)
        torque_history[-1] = current_torque
        delayed_torque = torque_history[0]
        data.ctrl[0] = delayed_torque
        mujoco.mj_step(model, data)

        time_steps.append(t)
        angles.append(angle)
        controls.append(delayed_torque)
        targets.append(current_target)

        if verbose:
            progress.update(1)

    if verbose:
        progress.close()

    return (np.array(controls), np.array(time_steps),
            np.array(angles), np.array(targets))