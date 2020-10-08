import unittest
import numpy as np
from particle_tracker_one_d import Trajectory


class AddTrajectoriesTester(unittest.TestCase):

    def test_addition_with_empty_trajectory(self):
        """
        Test adding one empty trajectory with a non empty trajectory
        """

        particle_positions = np.empty((1,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32)])

        particle_positions['frame_index'] = np.array([0])
        particle_positions['time'] = np.array([1])
        particle_positions['position'] = np.array([0])

        empty_trajectory = Trajectory(pixel_width=2)
        non_empty_trajectory = Trajectory(pixel_width=2)
        non_empty_trajectory.particle_positions = particle_positions

        final_trajectory = empty_trajectory + non_empty_trajectory

        np.testing.assert_array_equal(non_empty_trajectory.particle_positions, (non_empty_trajectory + empty_trajectory)._particle_positions)
        np.testing.assert_array_equal(non_empty_trajectory.particle_positions, (empty_trajectory + non_empty_trajectory)._particle_positions)
        np.testing.assert_array_equal(empty_trajectory.particle_positions, (empty_trajectory + empty_trajectory)._particle_positions)

    def test_addition_with_non_empty_trajectories(self):
        """
        Test adding together two trajectories
        """

        particle_positions_1 = np.empty((1,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32)])

        particle_positions_1['frame_index'] = np.array([0])
        particle_positions_1['time'] = np.array([1])
        particle_positions_1['position'] = np.array([0])

        particle_positions_2 = np.empty((1,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32)])

        particle_positions_2['frame_index'] = np.array([1])
        particle_positions_2['time'] = np.array([2])
        particle_positions_2['position'] = np.array([1])

        final_positions = np.empty((2,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32)])

        final_positions['frame_index'] = np.array([0, 1])
        final_positions['time'] = np.array([1, 2])
        final_positions['position'] = np.array([0, 1])

        t_1 = Trajectory()
        t_1.particle_positions = particle_positions_1
        t_2 = Trajectory()
        t_2.particle_positions = particle_positions_2

        np.testing.assert_array_equal(final_positions, (t_1 + t_2)._particle_positions)
        np.testing.assert_array_equal(final_positions, (t_2 + t_1)._particle_positions)

    def test_adding_overlapping_trajectories(self):
        """
        Test addition of two overlapping trajectories
        """

        particle_positions_1 = np.empty((3,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32)])

        particle_positions_1['frame_index'] = np.array([0, 1, 2])
        particle_positions_1['time'] = np.array([1, 2, 3])
        particle_positions_1['position'] = np.array([0, 3, 1])

        t_1 = Trajectory()
        t_1.particle_positions = particle_positions_1

        particle_positions_2 = np.empty((3,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32)])

        particle_positions_2['frame_index'] = np.array([1, 2, 3])
        particle_positions_2['time'] = np.array([2, 3, 4])
        particle_positions_2['position'] = np.array([5, 2, 0])

        t_2 = Trajectory()
        t_2.particle_positions = particle_positions_2

        final_positions = np.empty((4,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32)])

        final_positions['frame_index'] = np.array([0, 1, 2, 3])
        final_positions['time'] = np.array([1, 2, 3, 4])
        final_positions['position'] = np.array([0, 5, 2, 0])


        np.testing.assert_array_equal(final_positions, (t_1 + t_2)._particle_positions)
        np.testing.assert_array_equal(final_positions, (t_2 + t_1)._particle_positions)
