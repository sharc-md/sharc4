#!/usr/bin/env python3
from os import path
import numpy as np
from bisect import bisect_left

grids_path = path.dirname(__file__)
LEBEDEV_NPOINTS = {
    6: 3,
    14: 5,
    26: 7,
    38: 9,
    50: 11,
    74: 13,
    86: 15,
    110: 17,
    146: 19,
    170: 21,
    194: 23,
    230: 25,
    266: 27,
    302: 29,
    350: 31,
    434: 35,
    590: 41,
    770: 47,
    974: 53,
    1202: 59,
    1454: 65,
    1730: 71,
    2030: 77,
    2354: 83,
    2702: 89,
    3074: 95,
    3470: 101,
    3890: 107,
    4334: 113,
    4802: 119,
    5294: 125,
    5810: 131,
}

LEBEDEV_NPOINTS_k = sorted(LEBEDEV_NPOINTS.keys())


class LEBEDEV(object):
    """
    Class to hold the grid and prevent reloading in loop
    """
    grid = None

    def load(self, n_points: int) -> np.ndarray:
        """
        loads the according Lebedev grid with the given number of points as lower bound
        ------------
        Parameters:
        n_points: int Number of points for grid (lower bound)

        Returns:
        ndarray: shape=(>=n_points, 3) dtype=flaot
        """
        max_n = LEBEDEV_NPOINTS_k[-1]
        if 0 > n_points > max_n:
            raise ValueError(f'No Lebedev grid for {n_points} points!')
        # bisect search for in LEBEDEV_NPOINTS dict
        index = bisect_left(LEBEDEV_NPOINTS_k, n_points)
        n_points = LEBEDEV_NPOINTS_k[index]
        # load the closest (rounded upwards) grid from files
        if self.grid is None or len(self.grid) != n_points:
            degree = LEBEDEV_NPOINTS[n_points]
            self.grid = np.load(f'{grids_path}/lebedev_{degree}_{n_points}.npy')
            print(f'lebedev_{degree}_{n_points}')
        return self.grid


