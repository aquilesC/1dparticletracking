import unittest
import numpy as np
from particle_tracker_one_d import Trajectory
from setuptools import setup, find_packages


class TrajectoryTester(unittest.TestCase):
    class SimpleParticlePositionsExample:
        particle_positions = np.array([[0, 1], [1, 2], [2, 1], [3, 3], [4, 3]], dtype=np.int16)
        sample_time = 1
        boundaries = (0, 10)
        velocities = np.array([1, -1, 2, 0], dtype=np.float64)
        time_step = 1

    class ChangingTimeStepExample:
        particle_positions = np.array([[0, 1], [1, 2], [2, 3], [3, 5], [4, 3]], dtype=np.int16)
        time_step = 0.1
        velocities = np.array([10, 10, 20, -20])

    class ChangingPositionStepExample:
        particle_positions = np.array([[0, 1], [1, 2], [2, 3], [3, 5], [4, 3]], dtype=np.int16)
        position_step = 5e-6
        velocities = np.array([5e-6, 5e-6, 1e-5, -1e-5])

    class ChangingTimeAndPositionStepExample:
        particle_positions = np.array([[0, 1], [1, 2], [2, 3], [3, 5], [4, 3]], dtype=np.int16)
        position_step = 5e-6
        time_step = 0.1
        velocities = np.array([5e-5, 5e-5, 1e-4, -1e-4])

    class InitialiseMeanSquareDisplacementDictionaryExample:
        particle_positions = np.array([[0, 1], [1, 2], [2, 3], [3, 5], [4, 3], [6, 3], [20, 3]], dtype=np.int16)
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
        particle_positions = np.array([[0, 1], [1, 2], [2, 3], [3, 5], [4, 3]], dtype=np.int16)
        position_step = 1
        time_step = 1
        mean_square_displacements = np.array([2.5, 4.33333333, 8.5, 4], dtype=np.float32)
        mean_square_displacement_time = np.array([1, 2, 3, 4], dtype=np.float32)

    class CalculateMeanSquareDisplacementWithTimeAndPositionStepExample:
        particle_positions = np.array([[0, 1], [1, 2], [2, 3], [3, 5], [4, 3]], dtype=np.int16)
        position_step = 0.2
        time_step = 0.8
        mean_square_displacements = np.array([0.09999999, 0.17333335, 0.34, 0.16000003], dtype=np.float32)
        mean_square_displacement_time = np.array([0.8, 1.6, 2.4, 3.2], dtype=np.float32)

    class FitLineToMeanSquareDisplacementFunctionExample:
        particle_positions = np.array([[0, 1], [1, 2], [2, 3], [4, 5], [5, 6]], dtype=np.int16)
        expected_fit_coefficients = [6.0, -7.0]
        expected_errors = [0.6831300510639731, 2.2656860623955235]

    class CalculateDiffusionCoefficientExample:
        particle_positions = np.array([[0, 1], [1, 2], [2, 3], [4, 5], [5, 6]], dtype=np.int16)
        expected_diffusion_coefficient = 6.0 / 2
        expected_error = 2.2656860623955235 / 2

    class InfiniteDiffusionCoefficientExample:
        particle_positions = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [4, 3]], dtype=np.int16)
        position_step = 1
        time_step = 1
        diffusion_coefficient = np.Inf

    def test_append_particle_positions(self):
        trajectory = Trajectory(time_step=self.SimpleParticlePositionsExample.time_step)
        for position in self.SimpleParticlePositionsExample.particle_positions:
            trajectory.append_position(position)
        np.testing.assert_array_equal(trajectory._particle_positions, self.SimpleParticlePositionsExample.particle_positions)

    def test_particle_velocities(self):
        trajectory = Trajectory(time_step=self.SimpleParticlePositionsExample.time_step)
        for position in self.SimpleParticlePositionsExample.particle_positions:
            trajectory.append_position(position)
        trajectory._calculate_particle_velocities()
        np.testing.assert_array_equal(trajectory._velocities, self.SimpleParticlePositionsExample.velocities)

    def test_time_step(self):
        trajectory = Trajectory(time_step=self.ChangingTimeStepExample.time_step)
        for position in self.ChangingTimeStepExample.particle_positions:
            trajectory.append_position(position)
        trajectory._calculate_particle_velocities()
        np.testing.assert_array_equal(trajectory._velocities, self.ChangingTimeStepExample.velocities)

    def test_position_step(self):
        trajectory = Trajectory(position_step=self.ChangingPositionStepExample.position_step)
        for position in self.ChangingPositionStepExample.particle_positions:
            trajectory.append_position(position)
        trajectory._calculate_particle_velocities()
        np.testing.assert_array_equal(trajectory._velocities, self.ChangingPositionStepExample.velocities)

    def test_time_and_position_step(self):
        trajectory = Trajectory(position_step=self.ChangingTimeAndPositionStepExample.position_step, time_step=self.ChangingTimeAndPositionStepExample.time_step)
        for position in self.ChangingTimeAndPositionStepExample.particle_positions:
            trajectory.append_position(position)
        trajectory._calculate_particle_velocities()
        np.testing.assert_array_equal(trajectory._velocities, self.ChangingTimeAndPositionStepExample.velocities)

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
        trajectory.time_step = self.CalculateMeanSquareDisplacementWithTimeAndPositionStepExample.time_step
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
