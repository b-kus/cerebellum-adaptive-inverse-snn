# cerebellum-adaptive-inverse-snn
This is the capstone project for the course "Introduction to Neuromorphic Control". Spiking neural network (SNN) implementation of Kawato's Cerebellar Feedback Error Learning Model (CBFELM) for adaptive inverse control of 1 DoF pendulum simulated in Mujoco. 

## Overview
- **Goal:** Learn an inverse model of a 1-DOF pendulum online, compensating for a 50 ms transport delay.
- **Architecture:** 9 mossy fibers → 28 granule cells → 2 Purkinje cells (agonist-antagonist).
- **Learning:** LTD + homeostatic LTP driven by the PID error (climbing fibers).

## Result
The PD + SNN model reduces steady-state error by 67% compared to PD alone, demonstrating that the SNN can learn the gravity compensation torque online. However, the pendulum does not fully converge, and PID + SNN performs worse than PID alone (+37% error).

## How to run
1. clone the repo
2. install the requirements
3. run main.py
