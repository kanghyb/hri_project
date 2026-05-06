"""
display_project.py
------------------
Visual overlays in PyBullet: status sphere, theta bars, counters.
"""

import pybullet as p
import numpy as np

# Positions floating above and behind the table for the HUD
INDICATOR_POS    = [0.10, -0.45, 0.28]
LABEL_POS        = [0.16, -0.45, 0.28]
THETA_TITLE_POS  = [0.10, -0.45, 0.21]
THETA_COLOR_POS  = [0.10, -0.45, 0.16]
THETA_SIZE_POS   = [0.10, -0.45, 0.11]
THETA_SHAPE_POS  = [0.10, -0.45, 0.06]
CORR_POS         = [0.10, -0.45, 0.00]
TRIAL_POS        = [0.10, -0.45, -0.06]
ENTROPY_POS      = [0.10, -0.45, -0.12]

GREEN = [0.1, 0.9, 0.1]
RED   = [0.9, 0.1, 0.1]
AMBER = [1.0, 0.75, 0.1]
# Darker shades for readability against the bright sky / checkered floor
WHITE = [0.05, 0.05, 0.05]   # was [1, 1, 1] — now near-black so bars are visible
CYAN  = [0.0,  0.25, 0.65]   # was [0.3, 1, 1] — now a deep navy
GRAY  = [0.20, 0.20, 0.20]   # was [0.7, 0.7, 0.7] — now dark gray


def _bar(v, w=10):
    filled = int(round(v * w))
    return "|" + "#" * filled + "." * (w - filled) + f"| {v:.2f}"


class Display:

    def __init__(self):
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.025)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.025,
                                  rgbaColor=GREEN + [1.0])
        self._sphere = p.createMultiBody(baseMass=0,
                                          baseCollisionShapeIndex=col,
                                          baseVisualShapeIndex=vis,
                                          basePosition=INDICATOR_POS)

        self._lid = p.addUserDebugText("ROBOT CONTROL", LABEL_POS,
                                        textColorRGB=GREEN, textSize=1.3)
        self._tid = p.addUserDebugText("Learned theta:", THETA_TITLE_POS,
                                        textColorRGB=CYAN, textSize=0.9)
        self._bc  = p.addUserDebugText(f"  color  {_bar(1/3)}", THETA_COLOR_POS,
                                        textColorRGB=WHITE, textSize=0.85)
        self._bs  = p.addUserDebugText(f"  size   {_bar(1/3)}", THETA_SIZE_POS,
                                        textColorRGB=WHITE, textSize=0.85)
        self._bsh = p.addUserDebugText(f"  shape  {_bar(1/3)}", THETA_SHAPE_POS,
                                        textColorRGB=WHITE, textSize=0.85)
        self._cid = p.addUserDebugText("Corrections: 0", CORR_POS,
                                        textColorRGB=AMBER, textSize=0.9)
        self._trid = p.addUserDebugText("Trial 1 | Object 0/9", TRIAL_POS,
                                         textColorRGB=GRAY, textSize=0.85)
        self._eid  = p.addUserDebugText("Entropy: 1.099 (max)", ENTROPY_POS,
                                         textColorRGB=GRAY, textSize=0.85)

    def set_robot_mode(self):
        p.changeVisualShape(self._sphere, -1, rgbaColor=GREEN + [1.0])
        self._lid = p.addUserDebugText("ROBOT CONTROL", LABEL_POS,
                                        textColorRGB=GREEN, textSize=1.3,
                                        replaceItemUniqueId=self._lid)

    def set_human_mode(self):
        p.changeVisualShape(self._sphere, -1, rgbaColor=RED + [1.0])
        self._lid = p.addUserDebugText("HUMAN OVERRIDE", LABEL_POS,
                                        textColorRGB=RED, textSize=1.3,
                                        replaceItemUniqueId=self._lid)

    def set_window_mode(self):
        p.changeVisualShape(self._sphere, -1, rgbaColor=AMBER + [1.0])
        self._lid = p.addUserDebugText("WAITING 1/2/3...", LABEL_POS,
                                        textColorRGB=AMBER, textSize=1.2,
                                        replaceItemUniqueId=self._lid)

    def update_theta_bars(self, theta):
        self._bc  = p.addUserDebugText(f"  color  {_bar(theta[0])}", THETA_COLOR_POS,
                                        textColorRGB=WHITE, textSize=0.85,
                                        replaceItemUniqueId=self._bc)
        self._bs  = p.addUserDebugText(f"  size   {_bar(theta[1])}", THETA_SIZE_POS,
                                        textColorRGB=WHITE, textSize=0.85,
                                        replaceItemUniqueId=self._bs)
        self._bsh = p.addUserDebugText(f"  shape  {_bar(theta[2])}", THETA_SHAPE_POS,
                                        textColorRGB=WHITE, textSize=0.85,
                                        replaceItemUniqueId=self._bsh)

    def update_corrections(self, n):
        self._cid = p.addUserDebugText(f"Corrections: {n}", CORR_POS,
                                        textColorRGB=AMBER, textSize=0.9,
                                        replaceItemUniqueId=self._cid)

    def update_trial_info(self, trial, obj_idx, n_obj):
        self._trid = p.addUserDebugText(
            f"Trial {trial} | Object {obj_idx}/{n_obj}", TRIAL_POS,
            textColorRGB=GRAY, textSize=0.85,
            replaceItemUniqueId=self._trid)

    def update_entropy(self, entropy):
        max_h = np.log(3)
        pct   = entropy / max_h
        label = ("max uncertainty" if pct > 0.85 else
                 "uncertain"       if pct > 0.55 else
                 "converging"      if pct > 0.30 else
                 "confident!")
        self._eid = p.addUserDebugText(
            f"Entropy: {entropy:.3f} ({label})", ENTROPY_POS,
            textColorRGB=GRAY, textSize=0.85,
            replaceItemUniqueId=self._eid)