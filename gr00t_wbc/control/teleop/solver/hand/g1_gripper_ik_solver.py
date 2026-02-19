import numpy as np

from gr00t_wbc.control.teleop.solver.solver import Solver


######################################
### Define your solver here
######################################
class G1GripperInverseKinematicsSolver(Solver):
    def __init__(self, side) -> None:
        self.side = "L" if side.lower() == "left" else "R"

    def register_robot(self, robot):
        pass

    def __call__(self, finger_data):
        q_desired = np.zeros(7)

        # manus data
        fingertips = finger_data["position"]

        # Extract X, Y, Z positions of fingertips from the transformation matrices
        positions = np.array([finger[:3, 3] for finger in fingertips])

        # Ensure the positions are 2D arrays (N, 3)
        positions = np.reshape(positions, (-1, 3))  # Ensure 2D array with shape (N, 3)

        thumb_pos = positions[4, :]
        index_pos = positions[4 + 5, :]
        middle_pos = positions[4 + 10, :]
        ring_pos = positions[4 + 15, :]
        pinky_pos = positions[4 + 20, :]

        index_dist = np.linalg.norm(thumb_pos - index_pos)
        middle_dist = np.linalg.norm(thumb_pos - middle_pos)
        ring_dist = np.linalg.norm(thumb_pos - ring_pos)
        pinky_dist = np.linalg.norm(thumb_pos - pinky_pos)
        dist_threshold = 0.05

        index_close = index_dist < dist_threshold
        middle_close = middle_dist < dist_threshold
        ring_close = ring_dist < dist_threshold
        pinky_close = pinky_dist < dist_threshold

        if index_close:
            q_desired = self._get_index_close_q_desired()
        elif middle_close:
            q_desired = self._get_middle_close_q_desired()
        elif ring_close:
            q_desired = self._get_ring_close_q_desired()
        elif pinky_close:
            q_desired = self._get_pinky_close_q_desired()

        return q_desired

    def _get_index_close_q_desired(self):
        q_desired = np.zeros(7)

        amp0 = 0.5
        if self.side == "L":
            q_desired[0] -= amp0
        else:
            q_desired[0] += amp0

        amp = 0.7

        q_desired[1] += amp
        q_desired[2] += amp

        ampA1 = 1.5
        ampB1 = 1.5
        ampA2 = 0.6
        ampB2 = 1.5

        q_desired[3] -= ampA1
        q_desired[4] -= ampB1
        q_desired[5] -= ampA2
        q_desired[6] -= ampB2

        return q_desired if self.side == "L" else -q_desired

    def _get_middle_close_q_desired(self):
        q_desired = np.zeros(7)

        amp0 = 0.0
        if self.side == "L":
            q_desired[0] -= amp0
        else:
            q_desired[0] += amp0

        amp = 0.7

        q_desired[1] += amp
        q_desired[2] += amp

        ampA1 = 1.0
        ampB1 = 1.5
        ampA2 = 1.0
        ampB2 = 1.5

        q_desired[3] -= ampA1
        q_desired[4] -= ampB1
        q_desired[5] -= ampA2
        q_desired[6] -= ampB2

        return q_desired if self.side == "L" else -q_desired

    def _get_ring_close_q_desired(self):
        q_desired = np.zeros(7)

        amp0 = -0.5
        if self.side == "L":
            q_desired[0] -= amp0
        else:
            q_desired[0] += amp0

        amp = 0.7

        q_desired[1] += amp
        q_desired[2] += amp

        ampA1 = 0.6
        ampB1 = 1.5
        ampA2 = 1.5
        ampB2 = 1.5

        q_desired[3] -= ampA1
        q_desired[4] -= ampB1
        q_desired[5] -= ampA2
        q_desired[6] -= ampB2

        return q_desired if self.side == "L" else -q_desired

    def _get_pinky_close_q_desired(self):
        q_desired = np.zeros(7)

        return q_desired if self.side == "L" else -q_desired


class G1InspireInverseKinematicsSolver(Solver):
    """Simple heuristic hand IK that outputs 6 Inspire closure commands in [0, 1000]."""

    def __init__(self, side) -> None:
        self.side = "L" if side.lower() == "left" else "R"

    def register_robot(self, robot):
        pass

    @staticmethod
    def _closure_from_dist(dist: float, open_dist: float = 0.09, close_dist: float = 0.03) -> float:
        ratio = (open_dist - dist) / max(open_dist - close_dist, 1e-6)
        return float(np.clip(ratio, 0.0, 1.0))

    def __call__(self, finger_data):
        q_desired = np.zeros(6)

        fingertips = finger_data["position"]
        positions = np.array([finger[:3, 3] for finger in fingertips])
        positions = np.reshape(positions, (-1, 3))

        thumb_pos = positions[4, :]
        index_pos = positions[4 + 5, :]
        middle_pos = positions[4 + 10, :]
        ring_pos = positions[4 + 15, :]
        pinky_pos = positions[4 + 20, :]

        index_close = self._closure_from_dist(np.linalg.norm(thumb_pos - index_pos))
        middle_close = self._closure_from_dist(np.linalg.norm(thumb_pos - middle_pos))
        ring_close = self._closure_from_dist(np.linalg.norm(thumb_pos - ring_pos))
        pinky_close = self._closure_from_dist(np.linalg.norm(thumb_pos - pinky_pos))

        # Inspire command layout in this codebase:
        # [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
        thumb_close = max(index_close, middle_close, ring_close, pinky_close)
        q_desired[0] = 500.0 * thumb_close
        q_desired[1] = 1000.0 * thumb_close
        q_desired[2] = 1000.0 * index_close
        q_desired[3] = 1000.0 * middle_close
        q_desired[4] = 1000.0 * ring_close
        q_desired[5] = 1000.0 * pinky_close

        return q_desired
