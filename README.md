# cerebellum-adaptive-inverse-snn
Spiking neural network (SNN) implementation of Kawato's Cerebellar Feedback Error Learning Model (CBFELM) for adaptive inverse control of 1 DoF pendulum

## Overview
- **Goal:** Learn an inverse model of a 1-DOF pendulum online, compensating for a 50 ms transport delay.
- **Architecture:** 9 mossy fibers → 28 granule cells → 2 Purkinje cells (agonist-antagonist).
- **Learning:** LTD + homeostatic LTP driven by the PID error (climbing fibers).

## How to run
1. clone the repo
2. install the requirements
3. run main.py
