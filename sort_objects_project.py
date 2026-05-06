"""
sort_objects_project.py
-----------------------
SortObject class and PyBullet scene utilities.

TABLE SURFACE NOTE:
The pybullet table/table.urdf loaded at basePosition=[0.5, 0, -0.625]
puts the table surface at z = 0.0 in world coordinates.
Objects rest ON the table so their spawn z = half_extent + small_offset.
"""

import pybullet as p
import numpy as np
import random

COLORS = {
    "red":   [0.85, 0.15, 0.15, 1.0],
    "blue":  [0.15, 0.40, 0.85, 1.0],
    "green": [0.15, 0.65, 0.20, 1.0],
}
SIZES = {
    "small":  0.022,
    "medium": 0.030,
    "large":  0.038,
}
SHAPES = ["cube", "sphere", "cylinder"]

COLOR_IDX = {"red": 0, "blue": 1, "green": 2}
SIZE_IDX  = {"small": 0, "medium": 1, "large": 2}
SHAPE_IDX = {"cube": 0, "sphere": 1, "cylinder": 2}

# 3x3 grid on LEFT half of table (x, y only — z computed from size)
GRID_POSITIONS = [
    [0.32, -0.20], [0.47, -0.20], [0.62, -0.20],
    [0.32,  0.00], [0.47,  0.00], [0.62,  0.00],
    [0.32,  0.20], [0.47,  0.20], [0.62,  0.20],
]

# Bins on RIGHT half of table
BIN_XY = [
    [0.84, -0.25],
    [0.84,  0.00],
    [0.84,  0.25],
]

TABLE_Z       = 0.0
BIN_HOVER_Z   = 0.22
BIN_RELEASE_Z = 0.08
GRASP_Z_ABOVE = 0.20
GRASP_Z_AT    = 0.03


def get_hover_pos(bin_idx):
    xy = BIN_XY[bin_idx]
    return [xy[0], xy[1], BIN_HOVER_Z]


def get_release_pos(bin_idx):
    xy = BIN_XY[bin_idx]
    return [xy[0], xy[1], BIN_RELEASE_Z]


def bin_reward(obj, bin_idx, theta):
    """
    r = theta[0]*(color_correct_bin==bin) + theta[1]*(size_correct_bin==bin) + theta[2]*(shape_correct_bin==bin)
    High reward = this bin matches what the human cares about for this object.
    """
    r  = theta[0] * float(obj.correct_bin_by_color == bin_idx)
    r += theta[1] * float(obj.correct_bin_by_size  == bin_idx)
    r += theta[2] * float(obj.correct_bin_by_shape == bin_idx)
    return r


def all_bin_rewards(obj, theta):
    return np.array([bin_reward(obj, b, theta) for b in range(3)])


def correct_bin_under_theta(obj, theta):
    return int(np.argmax(all_bin_rewards(obj, theta)))


class SortObject:
    def __init__(self, color, size, shape, position_xy):
        self.color  = color
        self.size   = size
        self.shape  = shape
        self.pos_xy = position_xy
        self.body_id = None

        phi = np.zeros(9)
        phi[COLOR_IDX[color]]       = 1.0
        phi[3 + SIZE_IDX[size]]     = 1.0
        phi[6 + SHAPE_IDX[shape]]   = 1.0
        self.phi = phi

        self.correct_bin_by_color = COLOR_IDX[color]
        self.correct_bin_by_size  = SIZE_IDX[size]
        self.correct_bin_by_shape = SHAPE_IDX[shape]

    def spawn(self):
        half = SIZES[self.size]
        rgba = COLORS[self.color]
        z    = TABLE_Z + half + 0.003
        pos  = [self.pos_xy[0], self.pos_xy[1], z]

        if self.shape == "cube":
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half]*3)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half]*3, rgbaColor=rgba)
        elif self.shape == "sphere":
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=half)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=half, rgbaColor=rgba)
        else:
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=half, height=half*2)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=half, length=half*2, rgbaColor=rgba)

        self.body_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos)
        return self.body_id

    def get_world_position(self):
        if self.body_id is None:
            return [self.pos_xy[0], self.pos_xy[1], TABLE_Z + SIZES[self.size]]
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return list(pos)

    def respawn(self):
        """Remove and recreate at original grid position. Resets physics state."""
        self.remove()
        self.spawn()

    def remove(self):
        if self.body_id is not None:
            try:
                p.removeBody(self.body_id)
            except Exception:
                pass
            self.body_id = None

    def __repr__(self):
        return f"{self.color}-{self.size}-{self.shape}"


def make_object_set(n=9, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    colors    = list(COLORS.keys())
    sizes     = list(SIZES.keys())
    positions = random.sample(GRID_POSITIONS, n)
    objs = []
    for i, pos in enumerate(positions):
        color = colors[i % 3]
        size  = random.choice(sizes)
        shape = random.choice(SHAPES)
        objs.append(SortObject(color, size, shape, pos))
    random.shuffle(objs)
    return objs


def spawn_bins():
    # bigger bins: 18cm wide x 16cm deep x 8cm tall walls
    # each bin can comfortably hold 3 objects
    w, d, h, t = 0.18, 0.16, 0.10, 0.007
    bin_color  = [0.75, 0.75, 0.75, 0.8]
    labels     = ["Bin 1", "Bin 2", "Bin 3"]
    for i, bxy in enumerate(BIN_XY):
        bx, by = bxy
        walls = [
            # front and back walls
            ([bx,       by - d/2, h/2], [w/2, t/2, h/2]),
            ([bx,       by + d/2, h/2], [w/2, t/2, h/2]),
            # left and right walls
            ([bx - w/2, by,       h/2], [t/2, d/2, h/2]),
            ([bx + w/2, by,       h/2], [t/2, d/2, h/2]),
            # floor panel
            ([bx,       by,       t/2], [w/2, d/2, t/2]),
        ]
        for wpos, half in walls:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=bin_color)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                              baseVisualShapeIndex=vis, basePosition=wpos)
        # Place label on the FAR face of each bin — the +x side, opposite
        # the robot (robot sits at low x, shapes spawn between robot and bins).
        p.addUserDebugText(labels[i],
                           [bx + w/2 + 0.03, by, 0.04],
                           textColorRGB=[0.05, 0.05, 0.05], textSize=1.3)