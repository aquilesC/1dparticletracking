import unittest
import numpy as np
from particle_tracker_one_d import Trajectory


class CalculateDiffusionCoefficientTester(unittest.TestCase):

    def test_calculating_diffusion_coefficient_using_covariance_based_method(self):
        """
        Test that the calculation of diffusion coefficient from covariance based method is correct.
        """

        # Simplest test, diffusion coefficient = 0
        expected_diffusion_coefficient = 0.0

        particle_positions = np.empty((3,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32)])

        particle_positions['frame_index'] = [0, 1, 2]
        particle_positions['time'] = [0, 1, 2]
        particle_positions['position'] = [0, 0, 0]

        R = 1 / 4  # Representing exposure time equal to acquisition time

        pixel_width = 1

        t = Trajectory(pixel_width=pixel_width)

        t.particle_positions = particle_positions
        calculated_diffusion_coefficient, error = t.calculate_diffusion_coefficient_using_covariance_based_estimator(R=R)

        self.assertEqual(expected_diffusion_coefficient, calculated_diffusion_coefficient)


class PropertyTester(unittest.TestCase):

    def test_length_of_trajectory(self):
        """
        Test that the class property returns the length of the trajectory
        """

        particle_positions = np.empty((3,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32), ('zeroth_order_moment', np.float32),
                                                   ('second_order_moment', np.float32)])

        particle_positions['frame_index'] = [0, 1, 2]
        particle_positions['time'] = [0, 1, 2]
        particle_positions['position'] = [0, 0, 0]
        particle_positions['zeroth_order_moment'] = [0, 0, 0]
        particle_positions['second_order_moment'] = [0, 0, 0]

        t = Trajectory()

        expected_length_empty_trajectory = 0
        self.assertEqual(expected_length_empty_trajectory, t.length)

        t.particle_positions = particle_positions
        expected_length = 3
        self.assertEqual(expected_length, t.length)

    def test_density_of_trajectory(self):
        """
        Test that the density property returns the correct value.
        """

        t = Trajectory()

        expected_density_empty_trajectory = 1
        self.assertEqual(expected_density_empty_trajectory, t.density)

        particle_positions = np.empty((3,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32), ('zeroth_order_moment', np.float32),
                                                   ('second_order_moment', np.float32)])

        particle_positions['frame_index'] = [0, 1, 3]
        particle_positions['time'] = [0, 1, 3]
        particle_positions['position'] = [0, 0, 0]
        particle_positions['zeroth_order_moment'] = [0, 0, 0]
        particle_positions['second_order_moment'] = [0, 0, 0]

        t.particle_positions = particle_positions
        expected_density = 0.75
        self.assertEqual(expected_density, t.density)

        t.particle_positions['frame_index'] = [0, 1, 2]
        expected_density = 1
        self.assertEqual(expected_density, t.density)

        particle_positions = np.empty((5,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32), ('zeroth_order_moment', np.float32),
                                                   ('second_order_moment', np.float32)])

        particle_positions['frame_index'] = [1, 2, 3, 5, 7]
        particle_positions['time'] = [0, 1, 3, 4, 5]
        particle_positions['position'] = [0, 0, 0, 0, 0]
        particle_positions['zeroth_order_moment'] = [0, 0, 0, 0, 0]
        particle_positions['second_order_moment'] = [0, 0, 0, 0, 0]

        t.particle_positions = particle_positions
        expected_density = 0.7142857142857143
        self.assertEqual(expected_density, t.density)


class FunctionsTester(unittest.TestCase):

    def test_overlaps_function(self):
        t_1 = Trajectory()
        t_2 = Trajectory()

        self.assertTrue(not t_1.overlaps_with(t_2))
        self.assertTrue(not t_2.overlaps_with(t_1))

        particle_positions = np.empty((3,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32), ('zeroth_order_moment', np.float32),
                                                   ('second_order_moment', np.float32)])

        particle_positions['frame_index'] = [0, 1, 3]
        particle_positions['time'] = [0, 1, 3]
        particle_positions['position'] = [0, 0, 0]
        particle_positions['zeroth_order_moment'] = [0, 0, 0]
        particle_positions['second_order_moment'] = [0, 0, 0]

        t_1.particle_positions = particle_positions

        self.assertTrue(not t_1.overlaps_with(t_2))
        self.assertTrue(not t_2.overlaps_with(t_1))

        particle_positions = np.empty((3,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32), ('zeroth_order_moment', np.float32),
                                                   ('second_order_moment', np.float32)])

        particle_positions['frame_index'] = [0, 2, 3]
        particle_positions['time'] = [0, 1, 3]
        particle_positions['position'] = [0, 0, 0]
        particle_positions['zeroth_order_moment'] = [0, 0, 0]
        particle_positions['second_order_moment'] = [0, 0, 0]

        t_2.particle_positions = particle_positions

        self.assertTrue(t_1.overlaps_with(t_2))
        self.assertTrue(t_2.overlaps_with(t_1))

        particle_positions = np.empty((3,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32), ('zeroth_order_moment', np.float32),
                                                   ('second_order_moment', np.float32)])

        particle_positions['frame_index'] = [10, 11, 33]
        particle_positions['time'] = [0, 1, 3]
        particle_positions['position'] = [0, 0, 0]
        particle_positions['zeroth_order_moment'] = [0, 0, 0]
        particle_positions['second_order_moment'] = [0, 0, 0]

        t_2.particle_positions = particle_positions

        self.assertTrue(not t_1.overlaps_with(t_2))
        self.assertTrue(not t_2.overlaps_with(t_1))

    def test_split_trajectories_function(self):
        """
        Test that the splitting of two trajectories works.
        """

        t1 = Trajectory()
        t2 = Trajectory()

        particle_positions1 = np.empty((3,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32), ('zeroth_order_moment', np.float32),
                                                   ('second_order_moment', np.float32)])
        particle_positions1['frame_index'] = [0, 1, 2]
        particle_positions1['time'] = [0, 0, 0]
        particle_positions1['position'] = [0, 0, 0]
        particle_positions1['zeroth_order_moment'] = [0, 0, 0]
        particle_positions1['second_order_moment'] = [0, 0, 0]

        particle_positions2 = np.empty((4,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32), ('zeroth_order_moment', np.float32),
                                                    ('second_order_moment', np.float32)])
        particle_positions2['frame_index'] = [0, 2, 3, 4]
        particle_positions2['time'] = [0, 0, 0, 0]
        particle_positions2['position'] = [0, 0, 0, 0]
        particle_positions2['zeroth_order_moment'] = [0, 0, 0, 0]
        particle_positions2['second_order_moment'] = [0, 0, 0, 0]

        t1.particle_positions = particle_positions1
        t2.particle_positions = particle_positions2

        expected_trajectory_1 = Trajectory()
        expected_trajectory_2 = Trajectory()
        expected_trajectory_3 = Trajectory()
        expected_trajectory_4 = Trajectory()

        expected_trajectory_1.particle_positions = particle_positions1[:1]
        expected_trajectory_2.particle_positions = particle_positions1[1:2]
        expected_trajectory_3.particle_positions = particle_positions1[2:3]
        expected_trajectory_4.particle_positions = particle_positions2[2:]

        split_trajectories = t1.split(t2)

        np.testing.assert_array_equal(expected_trajectory_1.particle_positions, split_trajectories[0].particle_positions)
        np.testing.assert_array_equal(expected_trajectory_2.particle_positions, split_trajectories[1].particle_positions)
        np.testing.assert_array_equal(expected_trajectory_3.particle_positions, split_trajectories[2].particle_positions)
        np.testing.assert_array_equal(expected_trajectory_4.particle_positions, split_trajectories[3].particle_positions)


