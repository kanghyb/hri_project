"""
main_project.py
---------------
Online Bayesian IRL from Human Corrections.

SEQUENCE OF OPERATIONS:
1. Robot picks object using its current (wrong) policy -- sort by SIZE
2. Robot carries object to its chosen bin and PAUSES
3. Human model checks: is this the right bin under MY preference (color)?
4. If wrong: human overrides to correct bin, robot updates theta via IRL
5. If right: human accepts, no update
6. Over trials, robot theta converges from [0,1,0] toward [1,0,0]

Run: python main_project.py
"""

import pybullet as p
import pybullet_data
import numpy as np
import time

from robot               import Panda
from sort_objects_project import (make_object_set, spawn_bins,
                                   get_hover_pos, get_release_pos,
                                   correct_bin_under_theta, all_bin_rewards,
                                   TABLE_Z, SIZES)
from bayesian_irl_project import BayesianIRL
from human_model_project  import SimulatedHuman, RealHuman
from display_project      import Display

# ═══════════════════════ CONFIG ═══════════════════════════════════════
HUMAN_MODE = "simulated"   # "simulated" or "real"
N_TRIALS   = 5
N_OBJECTS  = 9

PERSONA = {
    "name":       "color",
    "theta_star": [1.0, 0.0, 0.0],   # human wants to sort by COLOR
    "epsilon":    0.0,
}

CONTROL_DT  = 1.0 / 240.0
JOINT_START = [0.0, 0.0, 0.0, -np.pi/2, 0.0, np.pi/2, np.pi/4,
               0.0, 0.0, 0.04, 0.04]

# Grasp parameters -- using ee_rotz=0 same as all HW scripts
# Table surface is at z=0.0.  Grasp z depends on object size.
GRASP_OFFSET = 0.01    # gripper descends to TABLE_Z + half_extent + offset
HOVER_ABOVE  = 0.22    # approach height above object before descending
# ══════════════════════════════════════════════════════════════════════

all_accuracies  = []
all_corrections = []


# ── logging helpers ───────────────────────────────────────────────────
def sep(c="=", w=66): print(c * w)
def hdr(t): sep(); print(f"  {t}"); sep()
def sub(t): sep("-"); print(f"  {t}"); sep("-")
def log(t): print(f"  {t}")
def blank(): print()


# ── arm motion ────────────────────────────────────────────────────────
# Joint limits and rest pose for constrained IK (biases toward elbow-forward)
_LL  = [-2.9,-1.76,-2.9,-3.07,-2.9,-0.02,-2.9]
_UL  = [ 2.9, 1.76, 2.9,-0.07, 2.9, 3.75, 2.9]
_JR  = [5.8, 3.5, 5.8, 3.0, 5.8, 3.8, 5.8]
_RP  = [0.0, -0.3, 0.0, -2.0, 0.0, 2.0, np.pi/4]  # elbow-forward rest pose

def _compute_ik(panda, pos, rotz):
    """
    Compute IK from a FIXED seed (JOINT_START) every time.
    
    PyBullet IK seeds from current joint state -- after any movement the
    solver finds a different (often elbow-backward) solution.  By temporarily
    resetting joints to JOINT_START before calling IK, we always get the
    same elbow-forward family of solutions regardless of where the arm
    physically is.  We immediately restore the real joint state so physics
    is not disturbed.
    """
    quat = p.getQuaternionFromEuler([np.pi, 0, rotz])

    # Save current joint state
    real_state = [p.getJointState(panda.panda, i)[0] for i in range(9)]

    # Temporarily set joints to rest pose for IK seeding
    for i, q in enumerate(_RP):
        p.resetJointState(panda.panda, i, q)

    joint_angles = p.calculateInverseKinematics(
        panda.panda, 11, pos, quat,
        lowerLimits=_LL, upperLimits=_UL,
        jointRanges=_JR, restPoses=_RP,
        maxNumIterations=200, residualThreshold=1e-5)

    # Restore real joint state so physics continues from where arm actually is
    for i, q in enumerate(real_state):
        p.resetJointState(panda.panda, i, q)

    # Log the IK solution so we can verify elbow-forward in terminal
    j = joint_angles
    log(f"  [IK]  seed=rest_pose  "
        f"j1={j[0]:.2f} j2={j[1]:.2f} j3={j[2]:.2f} j4={j[3]:.2f} "
        f"j5={j[4]:.2f} j6={j[5]:.2f} j7={j[6]:.2f}  "
        f"(j4 should be <0 for elbow-fwd, got {'OK' if j[3] < 0 else 'BAD-elbow-flip!'})")

    return joint_angles


def move(panda, pos, rotz=0.0, steps=500, gain=0.02):
    """
    Move to pos with ee_rotz=rotz.
    IK is solved from a fixed seed every step to prevent elbow-flip drift.
    """
    joint_angles = _compute_ik(panda, pos, rotz)   # compute once per waypoint
    for i in range(steps):
        p.setJointMotorControlArray(
            panda.panda, range(9), p.POSITION_CONTROL,
            targetPositions=joint_angles,
            positionGains=[gain]*9)
        p.stepSimulation()
        time.sleep(CONTROL_DT)
        if i % 100 == 0:
            st  = panda.get_state()
            ee  = st["ee-position"]
            dist = np.linalg.norm(np.array(ee) - np.array(pos))
            log(f"  [ARM]  ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})"
                f"  target=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})"
                f"  dist={dist:.3f}")
        p.stepSimulation()
        time.sleep(CONTROL_DT)
        if i % 100 == 0:
            st  = panda.get_state()
            ee  = st["ee-position"]
            dist = np.linalg.norm(np.array(ee) - np.array(pos))
            log(f"  [ARM]  ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})"
                f"  target=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})"
                f"  dist={dist:.3f}")


def settle(n=150):
    for _ in range(n):
        p.stepSimulation()
        time.sleep(CONTROL_DT)


def home(panda):
    log("[ARM]  returning home...")
    move(panda, [0.50, 0.0, 0.40], rotz=0.0, steps=400)


# ── accuracy ──────────────────────────────────────────────────────────
def compute_accuracy(placements, theta_star):
    correct = sum(1 for obj, actual in placements
                  if actual == correct_bin_under_theta(obj, theta_star))
    return correct / max(len(placements), 1)


# ── goal text on screen ───────────────────────────────────────────────
def add_goal_text(persona_name):
    # Darker, more saturated colors so the right-column text reads clearly
    # against the bright sky and white floor.
    ROBOT_COLOR = [0.00, 0.25, 0.70]   # deep blue (was light cyan)
    HUMAN_COLOR = [0.75, 0.25, 0.00]   # burnt orange (was light orange)
    LEGEND_COLOR = [0.15, 0.15, 0.15]  # near-black (was light gray)

    p.addUserDebugText(
        "ROBOT: starts sorting by SIZE (wrong!)",
        [0.30, 0.48, 0.50],
        textColorRGB=ROBOT_COLOR, textSize=1.1)
    p.addUserDebugText(
        "learns true preference from human corrections",
        [0.30, 0.48, 0.44],
        textColorRGB=ROBOT_COLOR, textSize=0.95)
    p.addUserDebugText(
        f"HUMAN: wants to sort by {persona_name.upper()}",
        [0.30, 0.48, 0.36],
        textColorRGB=HUMAN_COLOR, textSize=1.1)
    p.addUserDebugText(
        "overrides robot at the bin before release",
        [0.30, 0.48, 0.30],
        textColorRGB=HUMAN_COLOR, textSize=0.95)
    p.addUserDebugText("Bin 1 = red / small / cube",
                        [0.30, 0.48, 0.22],
                        textColorRGB=LEGEND_COLOR, textSize=0.85)
    p.addUserDebugText("Bin 2 = blue / medium / sphere",
                        [0.30, 0.48, 0.16],
                        textColorRGB=LEGEND_COLOR, textSize=0.85)
    p.addUserDebugText("Bin 3 = green / large / cylinder",
                        [0.30, 0.48, 0.10],
                        textColorRGB=LEGEND_COLOR, textSize=0.85)


# ── single trial ──────────────────────────────────────────────────────
def run_trial(trial_num, irl, human, display, panda, theta_star):

    irl.reset_trial()
    placements = []

    objects = make_object_set(n=N_OBJECTS, seed=trial_num * 7)
    for obj in objects:
        obj.spawn()
    settle(300)

    blank()
    hdr(f"TRIAL {trial_num}/{N_TRIALS}  |  PERSONA: {PERSONA['name']}"
        f"  |  theta*=[{theta_star[0]:.1f},{theta_star[1]:.1f},{theta_star[2]:.1f}]")

    theta   = irl.get_mean_theta()
    entropy = irl.get_entropy()
    log(f"Robot theta (mean) :  color={theta[0]:.3f}  size={theta[1]:.3f}  shape={theta[2]:.3f}")
    log(f"Posterior entropy  :  {entropy:.3f}")
    log(f"Total corrections  :  {irl.total_corrections}")
    blank()
    log("NOTE: Human model watches where robot carries each object.")
    log("      If the destination bin is WRONG under the human's preference,")
    log("      the human overrides BEFORE the robot releases the object.")
    blank()

    display.update_trial_info(trial_num, 0, N_OBJECTS)
    display.update_entropy(entropy)
    display.update_theta_bars(theta)

    for obj_idx, obj in enumerate(objects):

        sub(f"OBJECT {obj_idx+1}/{N_OBJECTS}  |  "
            f"{obj.color.upper()} . {obj.size.upper()} . {obj.shape.upper()}")

        log(f"Correct bin by color : Bin {obj.correct_bin_by_color+1}  ({obj.color})")
        log(f"Correct bin by size  : Bin {obj.correct_bin_by_size+1}  ({obj.size})")
        log(f"Correct bin by shape : Bin {obj.correct_bin_by_shape+1}  ({obj.shape})")
        log(f"Correct by theta*    : Bin {correct_bin_under_theta(obj,theta_star)+1}  (human's true goal)")

        display.update_trial_info(trial_num, obj_idx+1, N_OBJECTS)

        # robot decides
        robot_bin, theta_used, mode = irl.get_best_bin(obj)
        rewards = all_bin_rewards(obj, theta_used)
        blank()
        log(f"Robot reward values :  bin1={rewards[0]:.3f}  bin2={rewards[1]:.3f}  bin3={rewards[2]:.3f}")
        log(f"Robot decision      :  --> Bin {robot_bin+1}  ({mode})")
        log(f"Theta used          :  color={theta_used[0]:.3f}  size={theta_used[1]:.3f}  shape={theta_used[2]:.3f}")

        correct_for_human = correct_bin_under_theta(obj, theta_star)
        if robot_bin == correct_for_human:
            log(f"Robot choice matches human preference -- no override expected")
        else:
            log(f"Robot choice CONFLICTS with human -- override expected at bin")

        # get object position
        obj_pos = obj.get_world_position()
        half    = SIZES[obj.size]
        log(f"Object position : ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")

        # ── RESET JOINTS before each pick to prevent arm drift ────────
        # This snaps the arm back to a clean, known configuration so the
        # IK solver never gets stuck in a bad joint-space solution.
        log(f"[JOINT RESET]  snapping arm to home joint angles...")
        panda.reset(JOINT_START)
        settle(80)

        # ── PHASE 1: open gripper, move above object ──────────────────
        blank()
        log(f"[PHASE 1]  opening gripper, moving above object...")
        display.set_robot_mode()
        panda.open_gripper()
        above = [obj_pos[0], obj_pos[1], HOVER_ABOVE]
        move(panda, above, rotz=0.0, steps=450)

        ee = panda.get_state()["ee-position"]
        log(f"[PHASE 1 DONE]  ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})")

        # ── PHASE 2: descend to grasp height ─────────────────────────
        log(f"[PHASE 2]  descending to grasp height...")
        # Grasp at object center height: TABLE_Z + half_extent + small offset
        grasp_z = TABLE_Z + half + GRASP_OFFSET
        grasp_pos = [obj_pos[0], obj_pos[1], grasp_z]
        log(f"  grasp_z = {grasp_z:.4f}  (table={TABLE_Z} + half={half:.3f} + offset={GRASP_OFFSET})")
        move(panda, grasp_pos, rotz=0.0, steps=350, gain=0.015)

        ee = panda.get_state()["ee-position"]
        log(f"[PHASE 2 DONE]  ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})")

        # ── PHASE 3: grasp ────────────────────────────────────────────
        log(f"[PHASE 3]  closing gripper...")
        panda.close_gripper()
        settle(150)
        obj_after = obj.get_world_position()
        log(f"  object pos after grasp: ({obj_after[0]:.3f},{obj_after[1]:.3f},{obj_after[2]:.3f})")

        # ── PHASE 4: lift ─────────────────────────────────────────────
        log(f"[PHASE 4]  lifting to carry height...")
        carry_height = 0.25
        lift_pos = [obj_pos[0], obj_pos[1], carry_height]
        move(panda, lift_pos, rotz=0.0, steps=400)
        obj_lift = obj.get_world_position()
        lifted = obj_lift[2] > 0.08
        log(f"  object pos after lift: ({obj_lift[0]:.3f},{obj_lift[1]:.3f},{obj_lift[2]:.3f})")
        log(f"  lifted? {'YES' if lifted else 'NO -- grasp failed'}")

        # ── PHASE 5: carry to robot's chosen bin ─────────────────────
        log(f"[PHASE 5]  carrying to Bin {robot_bin+1} (robot's choice)...")
        hover = get_hover_pos(robot_bin)
        move(panda, hover, rotz=0.0, steps=600, gain=0.01)
        obj_carry = obj.get_world_position()
        log(f"  object pos at bin: ({obj_carry[0]:.3f},{obj_carry[1]:.3f},{obj_carry[2]:.3f})")

        # ── HUMAN CHECKS HERE ─────────────────────────────────────────
        blank()
        log(f"[OVERRIDE WINDOW]  robot hovering at Bin {robot_bin+1}...")
        log(f"  Human now checks: is Bin {robot_bin+1} correct for this {obj.color} object?")
        log(f"  Human's correct bin: Bin {correct_for_human+1}")

        if HUMAN_MODE == "real":
            log(f"  >>> Press 1/2/3 to redirect, or wait 2s to accept <<<")
        else:
            if robot_bin == correct_for_human:
                log(f"  Simulated human: Bin {robot_bin+1} is correct -- accepting")
            else:
                log(f"  Simulated human: Bin {robot_bin+1} is WRONG -- overriding to Bin {correct_for_human+1}")

        display.set_window_mode()
        human_bin, overrode = human.get_override(obj, robot_bin)

        if overrode:
            blank()
            log(f"*** HUMAN OVERRIDE ***")
            log(f"    Robot carried to : Bin {robot_bin+1}")
            log(f"    Human redirects  : Bin {human_bin+1}")
            log(f"    Reason           : robot sorts by {['color','size','shape'][int(np.argmax(theta_used))]}, human wants {PERSONA['name']}")

            display.set_human_mode()
            move(panda, get_hover_pos(human_bin), rotz=0.0, steps=400)

            old_theta   = irl.get_mean_theta().copy()
            old_entropy = irl.get_entropy()
            irl.update(robot_bin, human_bin, obj)
            new_theta   = irl.get_mean_theta()
            new_entropy = irl.get_entropy()

            blank()
            log(f"[IRL UPDATE #{irl.total_corrections}]")
            log(f"  correction    : Bin {robot_bin+1} (robot/size) --> Bin {human_bin+1} (human/color)")
            log(f"  old theta     : color={old_theta[0]:.3f}  size={old_theta[1]:.3f}  shape={old_theta[2]:.3f}")
            log(f"  new theta     : color={new_theta[0]:.3f}  size={new_theta[1]:.3f}  shape={new_theta[2]:.3f}")
            log(f"  entropy       : {old_entropy:.3f} --> {new_entropy:.3f}  "
                f"({'decreasing = learning!' if new_entropy < old_entropy else 'noisy'})")
            log(f"  dominant feat : {['color','size','shape'][int(np.argmax(new_theta))]}")

            display.update_theta_bars(new_theta)
            display.update_corrections(irl.total_corrections)
            display.update_entropy(new_entropy)
            actual_bin = human_bin

        else:
            log(f"[NO OVERRIDE]  human accepted Bin {robot_bin+1}.")
            log(f"  robot was correct -- no IRL update needed")
            display.set_robot_mode()
            actual_bin = robot_bin

        # ── PHASE 6: release ──────────────────────────────────────────
        log(f"[PHASE 6]  releasing into Bin {actual_bin+1}...")
        release = get_release_pos(actual_bin)
        #move(panda, release, rotz=0.0, steps=280, gain=0.015)
        panda.open_gripper()
        settle(120)

        obj_final = obj.get_world_position()
        correct   = correct_bin_under_theta(obj, theta_star)
        result    = "CORRECT" if actual_bin == correct else "WRONG"
        log(f"  object final pos: ({obj_final[0]:.3f},{obj_final[1]:.3f},{obj_final[2]:.3f})")
        log(f"[RESULT]  {obj} --> Bin {actual_bin+1}  (human wanted Bin {correct+1})  {result}")

        placements.append((obj, actual_bin))
        home(panda)
        display.set_robot_mode()
        blank()

    # cleanup
    for obj in objects:
        obj.remove()
    settle(50)

    accuracy  = compute_accuracy(placements, theta_star)
    theta     = irl.get_mean_theta()
    dist      = np.linalg.norm(theta - theta_star)
    correct_n = int(accuracy * N_OBJECTS)

    blank()
    hdr(f"TRIAL {trial_num} COMPLETE")
    log(f"Correct placements : {correct_n}/{N_OBJECTS}  ({accuracy*100:.1f}%)")
    log(f"Corrections        : {irl.corrections_this_trial}  (human intervened {irl.corrections_this_trial} times)")
    log(f"Total corrections  : {irl.total_corrections}")
    log(f"Robot theta (mean) : color={theta[0]:.3f}  size={theta[1]:.3f}  shape={theta[2]:.3f}")
    log(f"True  theta*       : color={theta_star[0]:.3f}  size={theta_star[1]:.3f}  shape={theta_star[2]:.3f}")
    log(f"Distance to theta* : {dist:.3f}  ({'converged!' if dist<0.25 else 'still learning'})")
    blank()

    # Clean up all objects from this trial (dropped, missed, or still on table)
    for obj in objects:
        obj.remove()
    settle(100)

    return accuracy


# ── main ──────────────────────────────────────────────────────────────
def main():

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.4, cameraYaw=50.0, cameraPitch=-32.0,
        cameraTargetPosition=[0.55, 0.0, 0.05])

    p.loadURDF("plane.urdf",       basePosition=[0, 0, -0.625])
    p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.625])
    spawn_bins()

    panda = Panda(basePosition=[0, 0, 0],
                  baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                  jointStartPositions=JOINT_START)
    panda.open_gripper()
    settle(200)

    irl     = BayesianIRL()
    display = Display()

    theta_star = np.array(PERSONA["theta_star"], dtype=float)
    theta_star /= theta_star.sum()

    human = (RealHuman() if HUMAN_MODE == "real"
             else SimulatedHuman(theta_star, PERSONA["epsilon"], PERSONA["name"]))

    add_goal_text(PERSONA["name"])

    blank()
    hdr("ONLINE BAYESIAN IRL FROM HUMAN CORRECTIONS")
    log("Human-Robot Interaction  |  Virginia Tech  |  Spring 2026")
    sep()
    log(f"Mode     : {HUMAN_MODE.upper()}")
    log(f"Persona  : {PERSONA['name']}")
    log(f"theta*   : color={theta_star[0]:.2f}  size={theta_star[1]:.2f}  shape={theta_star[2]:.2f}")
    sep()
    log("ROBOT PRIOR : sort by SIZE  (theta = [0.02, 0.96, 0.02])  <-- WRONG")
    log("HUMAN PREF  : sort by COLOR (theta* = [1.0, 0.0, 0.0])")
    log("HOW IT WORKS:")
    log("  1. Robot picks object, carries it to bin chosen by its current theta")
    log("  2. Robot PAUSES at the bin before releasing")
    log("  3. Human checks: is this the right bin under my preference?")
    log("  4. If WRONG: human overrides, robot runs IRL update, replans")
    log("  5. If RIGHT: robot releases, no update")
    log("EXPECTED: Trial 1 = ~6-7 overrides (robot sorts by size, human wants color)")
    log("          Trials 2-5 = fewer overrides as robot learns color preference")
    sep()
    if HUMAN_MODE == "real":
        log("CONTROLS: press 1/2/3 during WAITING window to redirect.")
        log("          No keypress within 2 seconds = accept robot's choice.")
    log("Starting in 3 seconds...")
    time.sleep(3)

    for trial in range(1, N_TRIALS + 1):
        acc = run_trial(trial, irl, human, display, panda, theta_star)
        all_accuracies.append(acc)
        all_corrections.append(irl.corrections_this_trial)

    blank()
    hdr("EXPERIMENT COMPLETE -- LEARNING SUMMARY")
    log(f"Final theta (mean) : {irl.get_mean_theta().round(3).tolist()}")
    log(f"True theta*        : {theta_star.round(3).tolist()}")
    log(f"Total corrections  : {irl.total_corrections}")
    blank()
    log("Trial | Accuracy | Corrections | Status")
    sep("-")
    for i, (acc, cor) in enumerate(zip(all_accuracies, all_corrections), 1):
        bar    = "#" * int(acc * 15)
        status = "still learning" if acc < 0.8 else ("converging" if acc < 1.0 else "converged!")
        log(f"  {i:2d}  |  {acc*100:5.1f}%  |  {cor:2d} overrides  |  {status}")
    blank()
    log(f"Trial 1 -> Trial 5 accuracy: "
        f"{all_accuracies[0]*100:.1f}% -> {all_accuracies[-1]*100:.1f}%")
    log(f"Total corrections to converge: {irl.total_corrections}")
    sep()
    log("Window open. Ctrl+C to exit.")

    while True:
        p.stepSimulation()
        time.sleep(CONTROL_DT)


if __name__ == "__main__":
    main()