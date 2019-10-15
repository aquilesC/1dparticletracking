import unittest
import numpy as np
from particle_tracker_one_d import Trajectory
from setuptools import setup, find_packages


class TrajectoryTester(unittest.TestCase):
    class SimpleParticlePositionsExample:
        particle_positions = np.empty((5,), dtype=[('frame_index', np.int16), ('time', np.float32), ('integer_position', np.int16), ('refined_position', np.float32)])
        particle_positions['frame_index'] = np.array([0, 1, 2, 3, 4])
        particle_positions['time'] = np.array([0, 1, 2, 3, 4]) * 0.5
        particle_positions['integer_position'] = np.array([1, 2, 1, 3, 3])
        particle_positions['refined_position'] = np.array([1.1, 2.3, 1.2, 2.9, 3.1])
        velocities = np.array([2.4, -2.2, 3.4, 0.4], dtype=np.float32)
        position_step = 1

    class ChangingPositionStepExample:
        particle_positions = np.empty((4,), dtype=[('frame_index', np.int16), ('time', np.float32), ('integer_position', np.int16), ('refined_position', np.float32)])
        particle_positions['frame_index'] = np.array([0, 1, 2, 3])
        particle_positions['time'] = np.array([0, 1, 2, 3]) * 0.5
        particle_positions['integer_position'] = np.array([1, 2, 3, 3])
        particle_positions['refined_position'] = np.array([1.1, 2.3, 2.9, 3.1])
        position_step = 0.1
        velocities = np.array([0.24, 0.12, 0.04])

    class InitialiseMeanSquareDisplacementDictionaryExample:
        particle_positions = np.empty((7,), dtype=[('frame_index', np.int16), ('time', np.float32), ('integer_position', np.int16), ('refined_position', np.float32)])
        particle_positions['frame_index'] = np.array([0, 1, 2, 3, 4, 6, 20])
        particle_positions['time'] = np.array([0, 1, 2, 3, 4, 6, 20]) * 0.5
        particle_positions['integer_position'] = np.array([1, 2, 3, 5, 3, 3, 3])
        particle_positions['refined_position'] = np.array([1, 2, 3, 5, 3, 3, 3]) + 0.2

        initial_mean_square_displacement_dictionary = {
            '1': None,
            '2': None,
            '3': None,
            '4': None,
            '5': None,
            '6': None,
            '14': None,
            '16': None,
            '17': None,
            '18': None,
            '19': None,
            '20': None
        }

    class CalculateMeanSquareDisplacementExample:
        particle_positions = np.empty((5,), dtype=[('frame_index', np.int16), ('time', np.float32), ('integer_position', np.int16), ('refined_position', np.float32)])
        particle_positions['frame_index'] = np.array([0, 1, 2, 3, 4])
        particle_positions['time'] = np.array([0, 1, 2, 3, 4]) * 0.5
        particle_positions['integer_position'] = np.array([1, 2, 3, 5, 3])
        particle_positions['refined_position'] = np.array([1, 2, 3, 5, 3]) + 0.2
        position_step = 1
        time_step = 1
        mean_square_displacements = np.array([2.5, 4.33333333, 8.5, 4], dtype=np.float32)
        mean_square_displacement_time = np.array([1, 2, 3, 4], dtype=np.float32) * 0.5

    class CalculateMeanSquareDisplacementWithTimeAndPositionStepExample:
        particle_positions = np.empty((5,), dtype=[('frame_index', np.int16), ('time', np.float32), ('integer_position', np.int16), ('refined_position', np.float32)])
        particle_positions['frame_index'] = np.array([0, 1, 2, 3, 4])
        particle_positions['time'] = np.array([0, 1, 2, 3, 4]) * 0.5
        particle_positions['integer_position'] = np.array([1, 2, 3, 5, 3])
        particle_positions['refined_position'] = np.array([1, 2, 3, 5, 3]) + 0.02
        position_step = 0.2
        mean_square_displacements = np.array([0.09999999, 0.17333329, 0.33999994, 0.15999998], dtype=np.float32)
        mean_square_displacement_time = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)

    class FitLineToMeanSquareDisplacementFunctionExample:
        particle_positions = np.empty((5,), dtype=[('frame_index', np.int16), ('time', np.float32), ('integer_position', np.int16), ('refined_position', np.float32)])
        particle_positions['frame_index'] = np.array([0, 1, 2, 4, 5])
        particle_positions['time'] = np.array([0, 1, 2, 4, 5])
        particle_positions['integer_position'] = np.array([1, 2, 3, 5, 6])
        particle_positions['refined_position'] = np.array([1, 2, 3, 5, 6])
        expected_fit_coefficients = [6.0, -7.0]
        expected_errors = [0.6831300510639731, 2.2656860623955235]

    class CalculateDiffusionCoefficientExample:
        particle_positions = np.empty((5,), dtype=[('frame_index', np.int16), ('time', np.float32), ('integer_position', np.int16), ('refined_position', np.float32)])
        particle_positions['frame_index'] = np.array([0, 1, 2, 4, 5])
        particle_positions['time'] = np.array([0, 1, 2, 4, 5])
        particle_positions['integer_position'] = np.array([1, 2, 3, 5, 6])
        particle_positions['refined_position'] = np.array([1, 2, 3, 5, 6])
        expected_diffusion_coefficient = 6.0 / 2
        expected_error = 2.2656860623955235 / 2

    class InfiniteDiffusionCoefficientExample:
        particle_positions = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [4, 3]], dtype=np.int16)
        position_step = 1
        time_step = 1
        diffusion_coefficient = np.Inf

    class DiffusionCoefficientFromCovarianceExample:
        particle_positions = np.empty((5,), dtype=[('frame_index', np.int16), ('time', np.float32), ('integer_position', np.int16), ('refined_position', np.float32)])
        particle_positions['frame_index'] = np.array([0, 1, 2, 3, 4])
        particle_positions['time'] = np.array([0, 1, 2, 3, 4]) * 0.8
        particle_positions['integer_position'] = np.array([1, 2, 3, 5, 3])
        particle_positions['refined_position'] = np.array([1, 2, 3, 5, 3])
        position_step = 0.2
        expected_diffusion_coefficient = 0.04583333836247547

    class SparseFrameIndexExample:
        particle_positions = np.empty((6,), dtype=[('frame_index', np.int16), ('time', np.float32), ('integer_position', np.int16), ('refined_position', np.float32)])
        particle_positions['frame_index'] = np.array([0, 2, 5, 6, 10, 11])
        particle_positions['time'] = np.array([0, 2, 5, 6, 10, 11]) * 0.8
        particle_positions['integer_position'] = np.array([1, 2, 3, 5, 3, 8])
        particle_positions['refined_position'] = np.array([1, 2, 3, 5, 3, 8])
        position_step = 0.2
        expected_diffusion_coefficient = 0.04583333836247547
        number_of_missing_data_points = 6
        number_of_particle_positions_with_single_time_step_between = 2


    def test_append_particle_positions(self):
        trajectory = Trajectory()
        for position in self.SimpleParticlePositionsExample.particle_positions:
            trajectory.append_position(position)
        np.testing.assert_array_equal(trajectory._particle_positions, self.SimpleParticlePositionsExample.particle_positions)

    def test_particle_velocities(self):
        trajectory = Trajectory()
        for position in self.SimpleParticlePositionsExample.particle_positions:
            trajectory.append_position(position)
        trajectory._calculate_particle_velocities()
        np.testing.assert_array_almost_equal(trajectory._velocities, self.SimpleParticlePositionsExample.velocities)

    def test_position_step(self):
        trajectory = Trajectory(position_step=self.ChangingPositionStepExample.position_step)
        for position in self.ChangingPositionStepExample.particle_positions:
            trajectory.append_position(position)
        trajectory._calculate_particle_velocities()
        np.testing.assert_array_almost_equal(trajectory._velocities, self.ChangingPositionStepExample.velocities)

    def test_initialise_dictionary_for_mean_square_displacement_function(self):
        trajectory = Trajectory()
        for position in self.InitialiseMeanSquareDisplacementDictionaryExample.particle_positions:
            trajectory.append_position(position)
        initial_mean_square_displacement_dictionary = trajectory._initialise_dictionary_for_mean_square_displacement_function()
        self.assertEqual(initial_mean_square_displacement_dictionary.keys(),
                         self.InitialiseMeanSquareDisplacementDictionaryExample.initial_mean_square_displacement_dictionary.keys())

    def test_calculate_mean_square_displacement_function(self):
        trajectory = Trajectory()
        for position in self.CalculateMeanSquareDisplacementExample.particle_positions:
            trajectory.append_position(position)
        trajectory._calculate_particle_velocities()
        time, mean_square_displacement = trajectory.calculate_mean_square_displacement_function()
        np.testing.assert_array_equal(time, self.CalculateMeanSquareDisplacementExample.mean_square_displacement_time)
        np.testing.assert_array_almost_equal(mean_square_displacement, self.CalculateMeanSquareDisplacementExample.mean_square_displacements)

    def test_calculate_mean_square_displacement_function_with_time_and_position_step(self):
        trajectory = Trajectory()
        for position in self.CalculateMeanSquareDisplacementWithTimeAndPositionStepExample.particle_positions:
            trajectory.append_position(position)
        trajectory.position_step = self.CalculateMeanSquareDisplacementWithTimeAndPositionStepExample.position_step
        trajectory._calculate_particle_velocities()
        time, mean_square_displacement = trajectory.calculate_mean_square_displacement_function()
        np.testing.assert_array_almost_equal(time, self.CalculateMeanSquareDisplacementWithTimeAndPositionStepExample.mean_square_displacement_time)
        np.testing.assert_array_almost_equal(mean_square_displacement, self.CalculateMeanSquareDisplacementWithTimeAndPositionStepExample.mean_square_displacements)

    def test_fit_a_straight_line_to_mean_square_displacement_function(self):
        trajectory = Trajectory()
        for position in self.FitLineToMeanSquareDisplacementFunctionExample.particle_positions:
            trajectory.append_position(position)
        trajectory._calculate_particle_velocities()
        fit_coefficients, error = trajectory._fit_straight_line_to_mean_square_displacement_function()
        self.assertAlmostEqual(fit_coefficients[0], self.FitLineToMeanSquareDisplacementFunctionExample.expected_fit_coefficients[0])
        self.assertAlmostEqual(fit_coefficients[1], self.FitLineToMeanSquareDisplacementFunctionExample.expected_fit_coefficients[1])
        self.assertAlmostEqual(error[0], self.FitLineToMeanSquareDisplacementFunctionExample.expected_errors[0])
        self.assertAlmostEqual(error[1], self.FitLineToMeanSquareDisplacementFunctionExample.expected_errors[1])

    def test_calculating_diffusion_coefficient_from_mean_square_displacement_function(self):
        trajectory = Trajectory()
        for position in self.CalculateDiffusionCoefficientExample.particle_positions:
            trajectory.append_position(position)
        trajectory._calculate_particle_velocities()
        diffusion_coefficient, error_estimate = trajectory.calculate_diffusion_coefficient_from_mean_square_displacement_function()
        self.assertAlmostEqual(diffusion_coefficient, self.CalculateDiffusionCoefficientExample.expected_diffusion_coefficient)

    def test_calculating_diffusion_coefficient_covariance_based_estimator(self):
        trajectory = Trajectory()
        for position in self.DiffusionCoefficientFromCovarianceExample.particle_positions:
            trajectory.append_position(position)
        trajectory.position_step = self.DiffusionCoefficientFromCovarianceExample.position_step
        diffusion_coefficient = trajectory.calculate_diffusion_coefficient_using_covariance_based_estimator()
        self.assertAlmostEqual(diffusion_coefficient, self.DiffusionCoefficientFromCovarianceExample.expected_diffusion_coefficient)

    def test_calculating_number_of_missing_data_points(self):
        trajectory = Trajectory()
        for position in self.SparseFrameIndexExample.particle_positions:
            trajectory.append_position(position)
        self.assertEqual(trajectory.calculate_number_of_missing_data_points(),self.SparseFrameIndexExample.number_of_missing_data_points)
        self.assertEqual(trajectory.calculate_number_of_particle_positions_with_single_time_step_between(),self.SparseFrameIndexExample.number_of_particle_positions_with_single_time_step_between)
