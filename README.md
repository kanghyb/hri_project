<iframe width="560" height="315" src="https://www.youtube.com/embed/yRGJsl7YXKE?si=b8hilrqJJ4JdE3K7" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

# Online Bayesian IRL from Human Corrections

A Panda robot arm sorts objects into bins. It starts with the wrong belief about what you want. It learns the right one — purely from your corrections.

Built for the Human-Robot Interaction course at Virginia Tech, Spring 2026.

---

## What it does

The robot sorts 9 objects (color × size × shape) into 3 bins across 5 trials. It starts convinced humans sort by **size**. The human's actual preference is **color**. Every time the robot pauses at the wrong bin and gets corrected, it runs a Bayesian update and gets a little smarter.

By trial 4, it sorts correctly with zero corrections needed.

---

## Setup

```bash
git clone https://github.com/vt-hri/HW4.git
cd HW4
python3 -m venv venv && source venv/bin/activate
pip install numpy pybullet
```

Copy the five project scripts into the `HW4/` folder alongside `robot.py`.

---

## Run

```bash
python main_project.py
```

The simulation opens in PyBullet. Watch the theta bars on screen shift from size-dominant to color-dominant across trials.

---

## Files

| File | Role |
|------|------|
| `main_project.py` | Main loop — grasping, IK, human override window |
| `bayesian_irl_project.py` | 300-particle Bayesian posterior update |
| `sort_objects_project.py` | Objects, bins, reward function |
| `human_model_project.py` | Simulated or keyboard human (press 1/2/3) |
| `display_project.py` | Live HUD — theta bars, correction counter |

---

## Key parameters

In `bayesian_irl_project.py`:

```python
BETA           = 2.0   # human rationality — higher = more deterministic
PRIOR_STRENGTH = 20.0  # learning speed dial — lower = faster, higher = slower
N_SAMPLES      = 300   # number of particles
N_MH_STEPS     = 40    # MH diversification steps per correction
```

---

## Reference

Losey et al., *Physical interaction as communication: Learning robot objectives online from human corrections*, IJRR 2022.

---

*Rohan Aaron Indupally & Bin Kang — Virginia Tech HRI Spring 2026*
