import unittest
import numpy as np
from source import Trajectory


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

    class CalculateMeanSquareDisplacementExample:
        particle_positions = np.array([[0, 1], [1, 2], [2, 3], [3, 5], [4, 3]], dtype=np.int16)
        position_step = 1
        time_step = 1
        mean_square_displacement = np.array([2.5, 4.3333335, 8.5])
        mean_square_displacement_time = np.array([1, 2, 3])

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

    def test_infinite_diffusion_coefficient(self):
        trajectory = Trajectory(time_step=self.InfiniteDiffusionCoefficientExample.time_step, position_step=self.InfiniteDiffusionCoefficientExample.position_step)
        for position in self.InfiniteDiffusionCoefficientExample.particle_positions:
            trajectory.append_position(position)
        trajectory._calculate_particle_velocities()
        time, mean_square_displacement = trajectory._calculate_mean_square_displacement_function()
        np.testing.assert_array_equal(time, self.CalculateMeanSquareDisplacementExample.mean_square_displacement_time)
        np.testing.assert_array_equal(mean_square_displacement, self.CalculateMeanSquareDisplacementExample.mean_square_displacement)
