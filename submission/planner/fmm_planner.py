from typing import List
import cv2
import skfmm
import numpy as np
from numpy import ma


class FMMPlanner:
    """
    Fast Marching Method Planner.
    """

    def __init__(self,
                 traversible: np.ndarray,
                 stop_distance: float,
                 scale: int = 1,
                 step_size: int = 5):
        """
        Arguments:
            traversible: (M + 1, M + 1) binary map encoding traversible regions
            stop_distance: distance to goal (in metres) under which to stop
            scale: map scale
            step_size: maximum distance of the short-term goal selected by the
             planner
        """
        self.stop_distance = stop_distance
        self.scale = scale
        self.step_size = step_size
        if scale != 1.:
            self.traversible = cv2.resize(traversible,
                                          (traversible.shape[1] // scale,
                                           traversible.shape[0] // scale),
                                          interpolation=cv2.INTER_NEAREST)
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.du = int(self.step_size / (self.scale * 1.))
        self.fmm_dist = None

    def set_multi_goal(self, goal_map: np.ndarray, visualize=False):
        """Set long-term goal(s) used to compute distance from a binary
        goal map.
        """
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        traversible_ma[goal_map == 1] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd

        if visualize:
            r, c = self.traversible.shape
            dist_vis = np.zeros((r, c * 3))
            dist_vis[:, :c] = self.traversible
            dist_vis[:, c:2 * c] = goal_map
            dist_vis[:, 2 * c:] = self.fmm_dist / self.fmm_dist.max()
            cv2.imshow("Planner Distance", dist_vis)
            cv2.waitKey(1)

    def get_short_term_goal(self, state: List[int]):
        """Compute the short-term goal closest to the current state.

        Arguments:
            state: current location
        """
        scale = self.scale * 1.
        state = [x / scale for x in state]
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        mask = FMMPlanner.get_mask(dx, dy, scale, self.step_size)
        dist_mask = FMMPlanner.get_dist(dx, dy, scale, self.step_size)

        state = [int(x) for x in state]

        dist = np.pad(self.fmm_dist, self.du,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)
        subset = dist[state[0]:state[0] + 2 * self.du + 1,
                      state[1]:state[1] + 2 * self.du + 1]

        assert subset.shape[0] == 2 * self.du + 1 and \
            subset.shape[1] == 2 * self.du + 1, \
            "Planning error: unexpected subset shape {}".format(subset.shape)

        subset *= mask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2

        # TODO Should we compute this as a function of the environment
        #  success distance (self.stop_distance)?
        if subset[self.du, self.du] < 1.0:
            stop = True
        else:
            stop = False

        subset -= subset[self.du, self.du]
        ratio1 = subset / dist_mask
        subset[ratio1 < -1.5] = 1

        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        if subset[stg_x, stg_y] > -0.0001:
            replan = True
        else:
            replan = False

        return (stg_x + state[0] - self.du) * scale, \
               (stg_y + state[1] - self.du) * scale, replan, stop

    @staticmethod
    def get_mask(sx, sy, scale, step_size):
        size = int(step_size // scale) * 2 + 1
        mask = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                cond1 = (((i + 0.5) - (size // 2 + sx)) ** 2 +
                         ((j + 0.5) - (size // 2 + sy)) ** 2) <= step_size ** 2
                cond2 = (((i + 0.5) - (size // 2 + sx)) ** 2 +
                         ((j + 0.5) - (size // 2 + sy)) ** 2) > (step_size - 1) ** 2
                if cond1 and cond2:
                    mask[i, j] = 1
        mask[size // 2, size // 2] = 1
        return mask

    @staticmethod
    def get_dist(sx, sy, scale, step_size):
        size = int(step_size // scale) * 2 + 1
        mask = np.zeros((size, size)) + 1e-10
        for i in range(size):
            for j in range(size):
                if (((i + 0.5) - (size // 2 + sx)) ** 2 +
                        ((j + 0.5) - (size // 2 + sy)) ** 2) <= step_size ** 2:
                    mask[i, j] = max(5,
                                     (((i + 0.5) - (size // 2 + sx)) ** 2 +
                                      ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
        return mask
