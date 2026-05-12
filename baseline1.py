"""
baseline1.py
------------
NO-LEARNING baseline.

The robot uses its initial size-biased prior  theta = [0.02, 0.96, 0.02]
for every decision and NEVER updates theta, regardless of how many
human corrections it receives.

Purpose: a flat ~33% accuracy line on the trial-vs-accuracy plot and a
single dot pinned at the SIZE corner on the simplex trajectory plot.
Without this baseline the IRL curve has nothing to look impressive
against.

Usage (full PyBullet visual demo, for video):
    python baseline1.py

Usage (fast headless run, for plot data only):
    set USE_PYBULLET = False below, then:
    python baseline1.py > baseline1.txt
"""

import numpy as np
from sort_objects_project import all_bin_rewards
from baseline_runner import run_demo

# ════════════════════════ CONFIG ════════════════════════════════════════
USE_PYBULLET = True    # True = full GUI demo (slow). False = headless (fast).
N_TRIALS     = 5       # 5 matches main_project. Set to 1 for a quick demo clip.
N_OBJECTS    = 9       # match main_project for apples-to-apples comparison

PERSONA = {
    "name":       "color",
    "theta_star": [1.0, 0.0, 0.0],
    "epsilon":    0.0,
}

ROBOT_PRIOR = np.array([0.02, 0.96, 0.02])  # frozen, never changes
# ════════════════════════════════════════════════════════════════════════


class NoLearning:
    """
    Static-theta brain. Implements the same interface as BayesianIRL but
    update() is a no-op: corrections are observed (counted) but theta
    never changes.
    """

    def __init__(self):
        self.theta = ROBOT_PRIOR.copy()
        self.total_corrections      = 0
        self.corrections_this_trial = 0

    def get_best_bin(self, obj):
        rewards = all_bin_rewards(obj, self.theta)
        return int(np.argmax(rewards)), self.theta.copy(), "no-learning (static)"

    def update(self, robot_bin, human_bin, obj):
        # observe but don't update
        self.total_corrections      += 1
        self.corrections_this_trial += 1

    def get_mean_theta(self):
        return self.theta.copy()

    def get_entropy(self):
        t = np.clip(self.theta, 1e-8, 1.0)
        t = t / t.sum()
        return float(-np.sum(t * np.log(t)))

    def reset_trial(self):
        self.corrections_this_trial = 0


if __name__ == "__main__":
    np.random.seed(0)
    intro = [
        "Method   : robot ignores all corrections, theta frozen",
        f"theta_0  : color={ROBOT_PRIOR[0]:.2f}  "
        f"size={ROBOT_PRIOR[1]:.2f}  shape={ROBOT_PRIOR[2]:.2f}  (frozen)",
        "EXPECTED : ~33% accuracy on every trial (chance level).",
        "           theta stays pinned at the SIZE corner of the simplex.",
    ]
    run_demo(NoLearning(),
             method_name="BASELINE 1 -- NO LEARNING",
             intro_lines=intro,
             n_trials=N_TRIALS,
             n_objects=N_OBJECTS,
             persona=PERSONA,
             use_pybullet=USE_PYBULLET)