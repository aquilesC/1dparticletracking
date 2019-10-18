import unittest
from particle_tracker_one_d import ParticleTracker
import numpy as np


def gaussian(amplitude=1, fwhm=1, mean=2, x=[0, 1, 2, 3, 4]):
    return amplitude * np.exp(-4 * np.log(2) * (x - mean) ** 2 / fwhm ** 2)


class ParticleTrackerTester(unittest.TestCase):
    class IntensityTwoParticlesCloseTooEachOtherExample:

        x_pixels = 100

        HWHM_real_particle = 20
        position_real_particle = 70
        amplitude_real_particle = 100

        HWHM_fake_particle = 4
        position_fake_particle = 50
        amplitude_fake_particle = 25

        intensity_real_particle = gaussian(amplitude=amplitude_real_particle, fwhm=2 * HWHM_real_particle, mean=position_real_particle, x=np.arange(0, x_pixels))
        intensity_fake_particle = gaussian(amplitude=amplitude_fake_particle, fwhm=2 * HWHM_fake_particle, mean=position_fake_particle, x=np.arange(0, x_pixels))

        time = [0]
        intensity = np.array([intensity_fake_particle + intensity_real_particle], dtype=np.uint16)
        intensity = ParticleTracker.normalise_intensity(intensity)

        number_of_particles_before_discrimination = 2
        number_of_particles_after_discrimination = 1

        initial_particle_positions_before_discrimination = np.array([[0, 60], [0, 70]])

    class IntensityExample:
        time = np.arange(0, 4)
        intensity = np.array([
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1],
        ])
        particle_positions = np.empty((7,), dtype=[('frame_index', np.int16), ('time', np.float32), ('integer_position', np.int16), ('refined_position', np.float32)])
        particle_positions['frame_index'] = np.array([0, 0, 1, 1, 2, 3, 3])
        particle_positions['time'] = np.array([0, 0, 1, 1, 2, 3, 3])
        particle_positions['integer_position'] = np.array([1, 3, 1, 4, 1, 1, 4])
        particle_positions['refined_position'] = np.array([1, 3, 1, 4, 1, 1, 4])

    class AssociationMatrixExample:
        time = np.array([0, 1, 2, 3])
        number_of_frames = 4
        number_of_future_frames_to_be_considered = 2
        maximum_distance_a_particle_can_travel_between_frames = 3
        particle_positions = np.empty((7,), dtype=[('frame_index', np.int16), ('time', np.float32), ('integer_position', np.int16), ('refined_position', np.float32)])
        particle_positions['frame_index'] = np.array([0, 0, 1, 1, 2, 3, 3])
        particle_positions['time'] = np.array([0, 0, 1, 1, 2, 3, 3])
        particle_positions['integer_position'] = np.array([1, 3, 1, 4, 1, 1, 4])
        particle_positions['refined_position'] = np.array([1, 3, 1, 4, 1, 1, 4])
        empty_association_matrix = {
            '0': {
                '1': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                '2': [[0, 0], [0, 0], [0, 0]]
            },
            '1': {
                '1': [[0, 0], [0, 0], [0, 0]],
                '2': [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            },
            '2': {
                '1': [[0, 0, 0], [0, 0, 0]],
            },
            '3': {}
        }

        initial_association_matrix = {
            '0': {
                '1': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                '2': [[1, 0], [0, 1], [1, 0]]
            },
            '1': {
                '1': [[1, 0], [0, 1], [1, 0]],
                '2': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            },
            '2': {
                '1': [[1, 0, 1], [0, 1, 0]],
            },
            '3': {}
        }

        trajectories = {
            '0': np.array([[0, 1], [1, 1], [2, 1], [3, 1]]),
            '1': np.array([[0, 3], [1, 4], [3, 4]])
        }

    class CenterOfMassExamples:
        number_of_data_points = 10
        x = np.arange(0, number_of_data_points)
        y_examples = [
            {
                'y': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                'center_of_mass': 4.5
            },
            {
                'y': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                'center_of_mass': 0
            },
            {
                'y': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                'center_of_mass': 9
            },
            {
                'y': np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
                'center_of_mass': 3
            },
        ]

    class FillInEmptyRowsAndColsExamples:
        empty_matrices = [
            np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ], dtype=np.int16),
            np.array([
                [0, 0],
                [0, 0],
                [0, 0]
            ], dtype=np.int16),
            np.array([
                [0, 0],
                [0, 1],
                [0, 0]
            ], dtype=np.int16)

        ]
        filled_matrices = [
            np.array([
                [1, 1, 1],
                [1, 0, 0],
                [1, 0, 0]
            ], dtype=np.int16),
            np.array([
                [1, 1],
                [1, 0],
                [1, 0]
            ], dtype=np.int16),
            np.array([
                [1, 0],
                [0, 1],
                [1, 0]
            ], dtype=np.int16)
        ]

    class TestFindingIntensityMaximasExample:
        intensity = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 0, 1],
            [0, 1, 0, 1, 0]
        ], dtype=np.int16)
        time = np.arange(0, intensity.shape[0]) * 0.5
        expected_positions = np.empty((9,), dtype=[('frame_index', np.int16), ('time', np.float32), ('integer_position', np.int16), ('refined_position', np.float32)])
        expected_positions['frame_index'] = np.array([0, 1, 2, 3, 4, 6, 6, 7, 7])
        expected_positions['time'] = np.array([0, 1, 2, 3, 4, 6, 6, 7, 7]) * 0.5
        expected_positions['integer_position'] = np.array([0, 1, 2, 3, 4, 2, 4, 1, 3])
        expected_positions['refined_position'] = np.array([0, 1, 2, 3, 4, 2, 4, 1, 3])

    class TestFindingParticlePositionExample:
        intensity = np.array([np.load('tests/intensity_test.npy')], dtype=np.float32)
        time = np.arange(0, intensity.shape[0])
        feature_point_threshold = 0.7
        expected_width_of_particle = 20
        expected_positions = np.empty((1,), dtype=[('frame_index', np.int16), ('time', np.float32), ('integer_position', np.int16), ('refined_position', np.float32)])
        expected_positions['frame_index'] = np.array([0])
        expected_positions['time'] = np.array([0])
        expected_positions['integer_position'] = np.array([108])
        expected_positions['refined_position'] = np.array([108.462814])

    class TestFindingNonIntegerParticlePositions:
        intensity = np.array([
            [0, 1, 2, 0, 0]
        ], dtype=np.int16)
        time = [0]
        feature_point_threshold = 1
        expected_width_of_particle = 2

        expected_positions = np.empty((1,), dtype=[('frame_index', np.int16), ('time', np.float32), ('integer_position', np.int16), ('refined_position', np.float32)])
        expected_positions['frame_index'] = np.array([0])
        expected_positions['time'] = np.array([0])
        expected_positions['integer_position'] = np.array([2])
        expected_positions['refined_position'] = np.array([5/3])


    def test_finding_particle_position(self):
        particle_tracker = ParticleTracker(frames=self.TestFindingParticlePositionExample.intensity, time=self.TestFindingParticlePositionExample.time)
        particle_tracker.feature_point_threshold = self.TestFindingParticlePositionExample.feature_point_threshold
        particle_tracker.integration_radius_of_intensity_peaks = self.TestFindingParticlePositionExample.expected_width_of_particle
        np.testing.assert_array_equal(self.TestFindingParticlePositionExample.expected_positions, particle_tracker.particle_positions)

    def test_finding_non_integer_particle_position(self):
        particle_tracker = ParticleTracker(frames=self.TestFindingNonIntegerParticlePositions.intensity, time=self.TestFindingNonIntegerParticlePositions.time)
        particle_tracker.integration_radius_of_intensity_peaks = self.TestFindingNonIntegerParticlePositions.expected_width_of_particle
        particle_tracker.feature_point_threshold = self.TestFindingNonIntegerParticlePositions.feature_point_threshold
        particle_tracker._update_particle_positions()
        print(particle_tracker.particle_positions)
        np.testing.assert_array_equal(particle_tracker.particle_positions, self.TestFindingNonIntegerParticlePositions.expected_positions)

    def test_finding_intensity_maximas(self):
        particle_tracker = ParticleTracker(frames=self.TestFindingIntensityMaximasExample.intensity, time=self.TestFindingIntensityMaximasExample.time)
        particle_tracker.feature_point_threshold = 0.6
        np.testing.assert_array_equal(particle_tracker._particle_positions, self.TestFindingIntensityMaximasExample.expected_positions)

    def test_center_of_mass_calculation(self):
        for example in self.CenterOfMassExamples.y_examples:
            self.assertEqual(example['center_of_mass'], ParticleTracker._calculate_center_of_mass(example['y']))

    def test_smoothing(self):
        particle_tracker = ParticleTracker(self.IntensityExample.intensity, self.IntensityExample.time)
        self.assertEqual(np.sum(particle_tracker.frames), np.sum(self.IntensityExample.intensity))

    def test_fill_in_empty_rows_and_cols(self):
        for index, link_matrix in enumerate(self.FillInEmptyRowsAndColsExamples.empty_matrices):
            actual_filled_matrix = ParticleTracker._fill_in_empty_rows_and_columns(link_matrix)
            np.testing.assert_array_equal(actual_filled_matrix, self.FillInEmptyRowsAndColsExamples.filled_matrices[index])

    def test_finding_particle_positions(self):
        particle_tracker = ParticleTracker(self.IntensityExample.intensity, self.IntensityExample.time)
        particle_tracker.feature_point_threshold = 0.2
        self.assertEqual(self.IntensityExample.particle_positions.shape, particle_tracker._particle_positions.shape)
        np.testing.assert_array_equal(particle_tracker._particle_positions, self.IntensityExample.particle_positions)

    def test_discrimnation_of_feature_points(self):
        particle_tracker = ParticleTracker(self.IntensityTwoParticlesCloseTooEachOtherExample.intensity, self.IntensityTwoParticlesCloseTooEachOtherExample.time)
        particle_tracker.feature_point_threshold = 0.3
        particle_tracker.integration_radius_of_intensity_peaks = 20
        particle_tracker.particle_discrimination_threshold = 0

        particle_tracker._find_integer_particle_positions()
        particle_tracker._refine_particle_positions()

        self.assertEqual(self.IntensityTwoParticlesCloseTooEachOtherExample.number_of_particles_before_discrimination, particle_tracker.particle_positions.shape[0])
        particle_tracker._perform_particle_discrimination()
        self.assertEqual(self.IntensityTwoParticlesCloseTooEachOtherExample.number_of_particles_after_discrimination, particle_tracker._particle_positions.shape[0])

    def test_initialising_association_matrix(self):
        particle_tracker = ParticleTracker(self.IntensityExample.intensity, self.IntensityExample.time)
        particle_tracker.integration_radius_of_intensity_peaks = 1
        particle_tracker.boxcar_width = 0
        particle_tracker.feature_point_threshold = 0.2
        particle_tracker.maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = 2
        particle_tracker.maximum_distance_a_particle_can_travel_between_frames = 2
        np.testing.assert_array_equal(particle_tracker._time, self.AssociationMatrixExample.time)
        np.testing.assert_array_equal(particle_tracker._particle_positions, self.AssociationMatrixExample.particle_positions)

        # Test keys in association matrix
        particle_tracker._initialise_empty_association_matrix()
        actual_empty_association_matrix = particle_tracker._association_matrix
        keys_in_actual_association_matrix = actual_empty_association_matrix.keys()
        keys_in_expected_association_matrix = self.AssociationMatrixExample.empty_association_matrix.keys()
        self.assertEqual(keys_in_expected_association_matrix, keys_in_actual_association_matrix,
                         msg='Time keys are not correct')
        for key in actual_empty_association_matrix.keys():
            self.assertEqual(self.AssociationMatrixExample.empty_association_matrix[key].keys(),
                             actual_empty_association_matrix[key].keys(), msg='Key: ' + key + ' is not correct')

        # Test the matrices in association matrix
        for time_key in actual_empty_association_matrix.keys():
            for r_key in actual_empty_association_matrix[time_key].keys():
                actual_matrix = actual_empty_association_matrix[time_key][r_key]
                expected_matrix = self.AssociationMatrixExample.empty_association_matrix[time_key][r_key]
                np.testing.assert_array_equal(actual_matrix, expected_matrix)

        # Test the initial linkings
        particle_tracker._update_association_matrix()
        actual_initital_association_matrix = particle_tracker.association_matrix.copy()
        for time_key in actual_empty_association_matrix.keys():
            for r_key in actual_initital_association_matrix[time_key].keys():
                actual_matrix = actual_initital_association_matrix[time_key][r_key]
                expected_matrix = self.AssociationMatrixExample.initial_association_matrix[time_key][r_key]
                np.testing.assert_array_equal(actual_matrix, expected_matrix, err_msg='time_key: ' + time_key + ', r_key: ' + r_key)

        # Test trajectories
        particle_tracker.future_frames_to_be_considered = 3
        particle_tracker._update_trajectories()
        print(particle_tracker.trajectories)
        number_of_actual_trajectories = len(particle_tracker.trajectories)
        expected_number_of_trajectories = len(self.AssociationMatrixExample.trajectories.keys())
        self.assertEqual(expected_number_of_trajectories, number_of_actual_trajectories)
