# cerebellum-adaptive-inverse-snn

This is the capstone project for the course **"Introduction to Neuromorphic Control"**. It implements a **spiking neural network (SNN)** based on **Kawato's Cerebellar Feedback Error Learning Model (CBFELM)** for adaptive inverse control of a 1-DoF pendulum simulated in MuJoCo.

The core idea is to use the error signal from a PID controller as a **teaching signal** (climbing fibres) to train a cerebellar-like SNN online, allowing it to learn the inverse dynamics of the pendulum and compensate for a 50 ms transport delay.

---

## Overview

- **Goal:** Learn an inverse model of a 1-DOF pendulum online, compensating for a 50 ms transport delay.
- **Architecture:** 9 mossy fibers → 28 granule cells → 2 Purkinje cells (agonist-antagonist).
- **Learning:** LTD + homeostatic LTP driven by the PID error (climbing fibers).
- **Key innovation:** Crossed climbing-fiber wiring to prevent the sign-inversion trap.

---

## Results

| Experiment       | Ki  | Baseline RMS | +SNN RMS | Change   |
|------------------|-----|--------------|----------|----------|
| PD vs PD+SNN     | 0   | 0.5599       | 0.1826   | **−67%** |
| PID vs PID+SNN   | 2.0 | 0.1660       | 0.2268   | **+37%** |

**Key takeaways:**
- The PD + SNN model reduces steady-state error by **67%** compared to PD alone, demonstrating that the SNN can learn the gravity compensation torque online.
- The pendulum **does not fully converge**, indicating that further tuning is needed.
- PID + SNN performs worse than PID alone (+37% error), suggesting that stochastic spiking activity interferes with the integral term.

---

## Repository Structure

```
.
├── main.py                # Entry point – runs experiments and generates figures
├── cerebellum/            # Core modules
│   ├── snn_model.py       # CerebellarSNN class (Brian2 network)
│   ├── controller.py      # PDController, OnlineSNNPIDController
│   └── pendulum_env.py    # run_pendulum(), pid_with_integral()
├── utils/                 # Helper functions
│   ├── encoding.py        # Gaussian population encoding
│   └── metrics.py         # RMS error, target function
├── config/                # Configuration
│   └── constants.py       # All global constants
├── assets/               # The XML file of the pendulum
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/b-kus/cerebellum-adaptive-inverse-snn.git
cd cerebellum-adaptive-inverse-snn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the main experiment

```bash
python main.py
```

This will:
- Run PD vs PD+SNN (Ki=0) for 20 seconds.
- Run PID vs PID+SNN (Ki=2.0) for 20 seconds.
- Save the figure to `figures/training_results.png`.

### 4. (Optional) Modify parameters

You can adjust the constants in `config/constants.py` to change training duration, PID gains, learning rates, or enable/disable the mid-run target perturbation.

---

## Dependencies

- Python 3.8+
- numpy
- brian2
- mujoco
- matplotlib
- tqdm

See `requirements.txt` for exact versions.

---

## Acknowledgments

This project was inspired by:
- Kawato & Gomi (1992) – A computational model of four regions of the cerebellum based on feedback-error learning.
- Wolpert, Miall, & Kawato (1998) – Internal models in the cerebellum.
- Vijayan & Diwakar (2022) – A cerebellum inspired spiking neural network for pattern classification and trajectory prediction.

---

