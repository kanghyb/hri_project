"""
human_model_project.py
-----------------------
Simulated and real human models.
"""

import numpy as np
import pybullet as p
import time
from sort_objects_project import correct_bin_under_theta

OVERRIDE_WINDOW = 2.0
CONTROL_DT      = 1.0 / 240.0


class SimulatedHuman:
    """
    Boltzmann-rational simulated human with hidden theta_star.
    Overrides robot when it places object in the wrong bin.
    """

    def __init__(self, theta_star, epsilon=0.0, name="human"):
        ts = np.array(theta_star, dtype=float)
        self.theta_star = ts / ts.sum()
        self.epsilon    = epsilon
        self.name       = name

    def get_correct_bin(self, obj):
        return correct_bin_under_theta(obj, self.theta_star)

    def get_override(self, obj, robot_bin):
        """
        Returns (human_bin, overrode: bool).
        """
        correct_bin = self.get_correct_bin(obj)

        if robot_bin == correct_bin:
            return robot_bin, False

        # human wants to override
        if np.random.rand() < self.epsilon:
            other = [b for b in [0, 1, 2] if b != robot_bin]
            return int(np.random.choice(other)), True
        else:
            return correct_bin, True


class RealHuman:
    """
    Real human presses 1/2/3 in the PyBullet window during override window.
    """

    def __init__(self):
        self.name = "real_human"

    def get_override(self, obj, robot_bin):
        start = time.time()
        while time.time() - start < OVERRIDE_WINDOW:
            keys = p.getKeyboardEvents()
            if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
                return 0, (0 != robot_bin)
            if ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
                return 1, (1 != robot_bin)
            if ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED:
                return 2, (2 != robot_bin)
            p.stepSimulation()
            time.sleep(CONTROL_DT)
        return robot_bin, False
