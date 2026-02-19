from gr00t_wbc.control.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
    G1InspireInverseKinematicsSolver,
)


# initialize hand ik solvers for g1 robot
def instantiate_g1_hand_ik_solver(hand_type: str = "dex3"):
    if hand_type in ("dex3", "three_finger"):
        solver_cls = G1GripperInverseKinematicsSolver
    elif hand_type == "inspire":
        solver_cls = G1InspireInverseKinematicsSolver
    else:
        raise ValueError(
            f"Unsupported hand_type '{hand_type}'. Expected one of "
            "['dex3', 'three_finger', 'inspire']."
        )

    left_hand_ik_solver = solver_cls(side="left")
    right_hand_ik_solver = solver_cls(side="right")
    return left_hand_ik_solver, right_hand_ik_solver
