from brian2 import *


# Timing constants
PHYS_DT = 0.005
CONTROL_EVERY = 4
CONTROL_DT = PHYS_DT * CONTROL_EVERY #every 20 ms
DELAY_STEPS = 10
TORQUE_LIMIT = 10.0
TRAIN_DURATION = 20.0
TARGET_ANGLE = 0.5
PERTURB_AT = 10.0
PERTURB_TARGET = 0.8

# Constants for Learning Algo 
DECODE_GAIN = 0.08
CF_SCALE = 2.0
ETA_LTD = 0.0008
LTP_RATE = 0.00002

KP = 8.0
KD = 1.5
KI_PD = 0.0
KI_PID = 2.0

# SNN constants
N_MOSSY = 9
N_GRANULE = 28
GC_FANIN = 4
MOSSY_MAX_RATE = 100 * Hz
CF_MAX_RATE = 100 * Hz
TAU_M = 10 * ms
V_THRESH = 1.0
V_RESET = 0.0
T_REF = 2 * ms
TAU_TRACE = 30 * ms
TAU_CF = 30 * ms
GAUSS_CENTERS = np.array([0.0, 0.5, 1.0])
GAUSS_SIGMA = 0.35

# ---------- MUJOCO XML ----------
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