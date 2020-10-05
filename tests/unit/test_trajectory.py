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

        R = 1/4 # Representing exposure time equal to acquisition time

        pixel_width = 1

        t = Trajectory(pixel_width=pixel_width)

        t._particle_positions = particle_positions
        calculated_diffusion_coefficient, error = t.calculate_diffusion_coefficient_using_covariance_based_estimator(R=R)

        self.assertEqual(expected_diffusion_coefficient, calculated_diffusion_coefficient)

class PropertyTester(unittest.TestCase):

    def test_length_of_trajectory(self):
        """
        Test that the class property returns the length of the trajectory
        """

        particle_positions = np.empty((3,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32), ('first_order_moment', np.float32), ('second_order_moment', np.float32)])

        particle_positions['frame_index'] = [0, 1, 2]
        particle_positions['time'] = [0, 1, 2]
        particle_positions['position'] = [0, 0, 0]
        particle_positions['first_order_moment'] = [0, 0, 0]
        particle_positions['second_order_moment'] = [0, 0, 0]

        t = Trajectory()

        expected_length_empty_trajectory = 0
        self.assertEqual(expected_length_empty_trajectory, t.length)

        t._particle_positions = particle_positions
        expected_length = 3
        self.assertEqual(expected_length, t.length)





