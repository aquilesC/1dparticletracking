import unittest
import numpy as np
from particle_tracker_one_d import ParticleTracker


class SetAttributeTester(unittest.TestCase):
    mock_frames = np.zeros((2, 3), dtype=np.float32)

    def test_validation_of_frames_argument(self):
        """
        Test that the frames are a numpy array with shape (nFrames,nPixels) with float values between 0 and 1.0.
        The number of frames nFrames should be larger than one and the nPixels should be larger than 2.
        """
        valid_frames = [
            np.array([[0, 1, 0], [0, 0, 0]], dtype=np.float32),
            np.array([[0.1, 0, 0.34], [0.1, 0, 1], [0.12, 0.003, 0.9]], dtype=np.float32),
        ]
        non_valid_shape_or_value_frames = [
            np.array([[0, 1, 0]], dtype=np.float32),
            np.array([0, 1, 0, 0.1], dtype=np.float32),
            np.array([[0.2], [0.4]], dtype=np.float32),
        ]
        non_valid_type_of_frames = [
            2,
            [[0, 1, 0], [0, 0, 0]],
            'string'
        ]
        for frames in valid_frames:
            self.assertTrue(ParticleTracker._test_if_frames_have_correct_format(frames), msg='Valid frames not accepted, frames: ' + np.array_str(frames))

        for frames in non_valid_shape_or_value_frames:
            with self.assertRaises(ValueError):
                ParticleTracker._test_if_frames_have_correct_format(frames)

        for frames in non_valid_type_of_frames:
            with self.assertRaises(TypeError):
                ParticleTracker._test_if_frames_have_correct_format(frames)

    def test_validation_time_argument(self):
        """
        Test that the validation of the class argument works. The time should be a numpy.ndarray with monotonically increasing values.
        """
        valid_times = [
            np.array([0, 1, 2, 4, 6], dtype=np.float32),
            np.array([0, 1000], dtype=np.float32),
            np.array([0.1, 0.2, 0.4], dtype=np.float32),
            np.array([1, 2, 3], dtype=np.float32)
        ]

        non_valid_shape_or_values_times = [
            np.array([0, 0], dtype=np.float32),
            np.array([0, -1], dtype=np.float32),
            np.array([[0, 2], [3, 5]], dtype=np.float32)
        ]

        non_valid_types_of_times = [
            '1,2,3,4',
            [1, 2, 3, 4],
            1,
            {'1': 1, '2': 2}
        ]

        for times in valid_times:
            self.assertTrue(ParticleTracker._test_if_time_has_correct_format(times), msg='Valid times not accepted, times: ' + np.array_str(times))

        for times in non_valid_shape_or_values_times:
            with self.assertRaises(ValueError):
                ParticleTracker._test_if_time_has_correct_format(times)

        for times in non_valid_types_of_times:
            with self.assertRaises(TypeError):
                ParticleTracker._test_if_time_has_correct_format(times)

    def test_validation_of_time_and_frames_having_same_length(self):
        """ Test that the validation of class arguments time and frames should have the same length."""
        valid_times_and_frames = [
            {
                'time': np.array([0, 1], dtype=np.float32),
                'frames': np.array([[0, 1, 0], [0, 0, 0]], dtype=np.float32)
            },
            {
                'time': np.array([0.1, 0.2, 0.4], dtype=np.float32),
                'frames': np.array([[0.1, 0, 0.34], [0.1, 0, 1], [0.12, 0.003, 0.9]], dtype=np.float32)
            }
        ]

        non_valid_times_and_frames = [
            {
                'time': np.array([0, 1], dtype=np.float32),
                'frames': np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0.4]], dtype=np.float32)
            },
            {
                'time': np.array([0.1, 0.2, 0.4], dtype=np.float32),
                'frames': np.array([[0.1, 0, 0.34], [0.1, 0, 1]], dtype=np.float32)
            }
        ]

        for time_and_frames in valid_times_and_frames:
            self.assertTrue(ParticleTracker._test_if_time_and_frames_has_same_length(time_and_frames['time'], time_and_frames['frames']))

        for time_and_frames in non_valid_times_and_frames:
            with self.assertRaises(ValueError):
                ParticleTracker._test_if_time_and_frames_has_same_length(time_and_frames['time'], time_and_frames['frames'])

    def test_validation_of_automatic_update(self):

        valid_automatic_update_arguments = [True, False]

        non_valid_automatic_update_arguments = [0, 1, 'True', 'False', np.array, []]

        for automatic_update in valid_automatic_update_arguments:
            self.assertTrue(ParticleTracker._test_if_automatic_update_has_correct_format(automatic_update))

        for automatic_update in non_valid_automatic_update_arguments:
            with self.assertRaises(ValueError):
                ParticleTracker._test_if_automatic_update_has_correct_format(automatic_update)

    def test_validation_of_setting_the_integration_radius_of_intensity_peaks(self):
        """
        Tests the setting of the class attribute integration_radius_of_intensity_peaks. Should be an integer smaller than half of the number of pixels in a frame.
        """
        frames = np.array([
            [0, 0.1, 0.2, 0.1],
            [0, 0.2, 0.3, 0.4],
            [0.2, 0.5, 0.6, 1],
            [0, 0.1, 0.2, 0.1]
        ], dtype=np.float32)
        time = np.array([0, 1, 2, 3])
        automatic_update = False

        valid_integration_radius = [0, 1, 2]
        non_valid_type_of_integration_radius = [1.5, '1', [1, 2]]
        non_valid_values_of_integration_radius = [-1, 3, 100]

        pt = ParticleTracker(frames=frames, time=time, automatic_update=automatic_update)

        for radius in valid_integration_radius:
            pt.integration_radius_of_intensity_peaks = radius
            self.assertEqual(pt.integration_radius_of_intensity_peaks, radius)

        for radius in non_valid_type_of_integration_radius:
            with self.assertRaises(TypeError, msg=radius):
                pt.integration_radius_of_intensity_peaks = radius

        for radius in non_valid_values_of_integration_radius:
            with self.assertRaises(ValueError, msg=radius):
                pt.integration_radius_of_intensity_peaks = radius

    def test_validation_of_setting_boxcar_width(self):
        """
        Tests the setting of the class attribute boxcar_width. Should be an integer smaller than the number of pixels in a frame.
        """
        frames = np.array([
            [0, 0.1, 0.2, 0.1],
            [0, 0.2, 0.3, 0.4],
            [0.2, 0.5, 0.6, 1],
            [0, 0.1, 0.2, 0.1]
        ], dtype=np.float32)
        time = np.array([0, 1, 2, 3])
        automatic_update = False

        valid_boxcar_widths = [0, 1, 2, 3, 4]
        non_valid_type_of_boxcar_widths = [1.5, '1', [1, 2]]
        non_valid_values_of_boxcar_widths = [-1, 5, 100]

        pt = ParticleTracker(frames=frames, time=time, automatic_update=automatic_update)

        for width in valid_boxcar_widths:
            pt.boxcar_width = width
            self.assertEqual(pt.boxcar_width, width)

        for width in non_valid_type_of_boxcar_widths:
            with self.assertRaises(TypeError, msg=width):
                pt.boxcar_width = width

        for width in non_valid_values_of_boxcar_widths:
            with self.assertRaises(ValueError, msg=width):
                pt.boxcar_width = width

    def test_validation_of_setting_particle_detection_threshold(self):
        """
        Tests the setting of the class attribute particle_detection_threshold. Should be a numerical value between 0 and 1.
        """
        frames = np.array([
            [0, 0.1, 0.2, 0.1],
            [0, 0.2, 0.3, 0.4],
            [0.2, 0.5, 0.6, 1],
            [0, 0.1, 0.2, 0.1]
        ], dtype=np.float32)
        time = np.array([0, 1, 2, 3])
        automatic_update = False

        valid_particle_detection_thresholds = [0, 0.1, 0.22, 0.93, 1]
        non_valid_type_detection_thresholds = ['1', [1, 2], None]
        non_valid_values_of_detection_thresholds = [-1, 5, 100]

        pt = ParticleTracker(frames=frames, time=time, automatic_update=automatic_update)

        for threshold in valid_particle_detection_thresholds:
            pt.particle_detection_threshold = threshold
            self.assertEqual(pt.particle_detection_threshold, threshold)

        for threshold in non_valid_type_detection_thresholds:
            with self.assertRaises(TypeError, msg=threshold):
                pt.particle_detection_threshold = threshold

        for threshold in non_valid_values_of_detection_thresholds:
            with self.assertRaises(ValueError, msg=threshold):
                pt.particle_detection_threshold = threshold

    def test_validation_of_setting_maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles(self):
        """
        Tests the setting of the class attribute maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles. Should be a numerical value between 0 and 1.
        """
        frames = np.array([
            [0, 0.1, 0.2, 0.1],
            [0, 0.2, 0.3, 0.4],
            [0.2, 0.5, 0.6, 1],
            [0, 0.1, 0.2, 0.1]
        ], dtype=np.float32)
        time = np.array([0, 1, 2, 3])
        automatic_update = False

        valid_maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = [0, 1, 2, 3]
        non_valid_type_maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = ['1', [1, 2], None, 1.2]
        non_valid_values_of_maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = [-1, 5, 100]

        pt = ParticleTracker(frames=frames, time=time, automatic_update=automatic_update)

        for number in valid_maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles:
            pt.maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = number
            self.assertEqual(pt.maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles, number)

        for number in non_valid_type_maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles:
            with self.assertRaises(TypeError, msg=number):
                pt.maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = number

        for number in non_valid_values_of_maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles:
            with self.assertRaises(ValueError, msg=number):
                pt.maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = number

    def test_validation_of_setting_maximum_distance_a_particle_can_travel_between_frames(self):
        """
        Tests the setting of the class attribute maximum_distance_a_particle_can_travel_between_frames. Should be a numerical value between 0 and number of pixels in each frame.
        """
        frames = np.array([
            [0, 0.1, 0.2, 0.1],
            [0, 0.2, 0.3, 0.4],
            [0.2, 0.5, 0.6, 1],
            [0, 0.1, 0.2, 0.1]
        ], dtype=np.float32)
        time = np.array([0, 1, 2, 3])
        automatic_update = False

        valid_maximum_distance_a_particle_can_travel_between_frames = [0.4, 1.4, 2, 3]
        non_valid_type_maximum_distance_a_particle_can_travel_between_frames = ['1', [1, 2], None]
        non_valid_values_of_maximum_distance_a_particle_can_travel_between_frames = [-1, 5, 100]

        pt = ParticleTracker(frames=frames, time=time, automatic_update=automatic_update)

        for distance in valid_maximum_distance_a_particle_can_travel_between_frames:
            pt.maximum_distance_a_particle_can_travel_between_frames = distance
            self.assertEqual(pt.maximum_distance_a_particle_can_travel_between_frames, distance)

        for distance in non_valid_type_maximum_distance_a_particle_can_travel_between_frames:
            with self.assertRaises(TypeError, msg=distance):
                pt.maximum_distance_a_particle_can_travel_between_frames = distance

        for distance in non_valid_values_of_maximum_distance_a_particle_can_travel_between_frames:
            with self.assertRaises(ValueError, msg=distance):
                pt.maximum_distance_a_particle_can_travel_between_frames = distance

    def test_changing_the_cost_coefficients(self):
        """
        Test that it is possible to change cost coefficients
        """
        frames = np.array([
            [0, 0.1, 0.2, 0.1],
            [0, 0.2, 0.3, 0.4],
            [0.2, 0.5, 0.6, 1],
            [0, 0.1, 0.2, 0.1]
        ], dtype=np.float32)
        time = np.array([0, 1, 2, 3])
        automatic_update = False

        correct_cost_coefficients = np.array([
            [1, 2, 3],
            [0, 0.2, 0.1],
            [10, 10.2, -1]
        ])

        incorrect_cost_coefficients = np.array([
            [0, 0, 0]
        ])

        pt = ParticleTracker(frames=frames, time=time, automatic_update=automatic_update)

        for cost_coefficients in correct_cost_coefficients:
            pt.change_cost_coefficients(cost_coefficients[0], cost_coefficients[1], cost_coefficients[2])
            np.testing.assert_array_almost_equal(cost_coefficients, pt._cost_coefficients)

        for cost_coefficients in incorrect_cost_coefficients:
            with self.assertRaises(ValueError):
                pt.change_cost_coefficients(cost_coefficients[0], cost_coefficients[1], cost_coefficients[2])


class FindParticlePositionsTester(unittest.TestCase):

    def test_finding_local_maximas_function(self):
        """
        Tests finding particle positions. All local maximas should be found, that is, the value should be higher than both the neighbouring value to left and right.
        """
        intensity_examples = [
            np.array([1, 0, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
            np.array([0, 0, 1], dtype=np.float32),
            np.array([0, 0, 0], dtype=np.float32),
            np.array([0, 1, 1], dtype=np.float32),
            np.array([1, 1, 0], dtype=np.float32),
            np.array([1, 1, 1], dtype=np.float32),
            np.array([1, 0, 1], dtype=np.float32),
            np.array([1, 0, 1, 0], dtype=np.float32),
            np.array([1, 1, 0, 0], dtype=np.float32),
            np.array([1, 0, 0, 1], dtype=np.float32),
            np.array([0, 0, 1, 0], dtype=np.float32),
            np.array([0, 0, 0.3, 0], dtype=np.float32),
        ]
        expected_positions = [
            np.array([], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([2], dtype=np.int64)
        ]

        frames_examples = np.array([
            [0, 0.1, 0.5],
            [0, 0.6, 0.2],
            [1, 0.1, 0.1],
            [0.1, 0.1, 0.2],
            [0.7, 0.2, 0.7]
        ], dtype=np.float32)

        times_examples = np.array([0, 1, 2, 3, 4])

        pt = ParticleTracker(frames=frames_examples, time=times_examples, automatic_update=False)
        pt.particle_detection_threshold = 0.1

        for index, intensity in enumerate(intensity_examples):
            np.testing.assert_array_equal(expected_positions[index], pt._find_local_maximas_larger_than_threshold(intensity))

    def test_finding_initial_particle_positions_function(self):
        """
        Tests finding the initial particle positions before any discrimination and refinement of positions.
        """
        frames_examples = np.array([
            [0, 0.1, 0.5],
            [0, 0.6, 0.2],
            [1, 0.1, 0.1],
            [0.1, 0.1, 0.2],
            [0.7, 0.2, 0.7]
        ], dtype=np.float32)

        times_examples = np.array([0, 1, 2, 3, 4])

        expected_positions = [
            np.array([], dtype=np.float32),
            np.array([1], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32)
        ]

        automatic_update = False

        pt = ParticleTracker(time=times_examples, frames=frames_examples, automatic_update=automatic_update)
        pt.particle_detection_threshold = 0.3
        pt._find_initial_particle_positions()

        self.assertEqual(len(pt.particle_positions), len(expected_positions))

        for index, position in enumerate(pt.particle_positions):
            np.testing.assert_array_equal(expected_positions[index], position)

    def test_find_center_of_mass_function(self):
        """
        Test finding the center of mass close to the initial particle positions.
        """

        intensity_examples = [
            np.array([1, 0, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
            np.array([0, 0, 1], dtype=np.float32),
            np.array([1, 1, 0], dtype=np.float32)
        ]

        expected_center_of_mass = np.array([0, 1, 2, 0.5], dtype=np.float32)

        for index, intensity in enumerate(intensity_examples):
            self.assertEqual(ParticleTracker._calculate_center_of_mass(intensity), expected_center_of_mass[index])

    def test_the_refining_of_particle_positions(self):
        """
        Test that the refinement of particle positions work. That is, by finding the center of mass close to the initial particle position.
        """
        automatic_update = False

        frames_example = np.array([
            [0, 0.1, 0.5, 0],
            [0, 0.6, 0.2, 0],
            [1, 0.1, 0.1, 0],
            [0.1, 0.1, 0.2, 0],
            [0, 0.7, 0.2, 0.7]
        ], dtype=np.float32)

        times_example = np.array([0, 1, 2, 3, 4])

        expected_positions = [
            np.array([1.8333333], dtype=np.float32),
            np.array([1.25], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([1.2222222], dtype=np.float32)
        ]

        pt = ParticleTracker(time=times_example, frames=frames_example, automatic_update=automatic_update)
        pt.particle_detection_threshold = 0.3
        pt.integration_radius_of_intensity_peaks = 1
        pt._find_initial_particle_positions()
        pt._refine_particle_positions()

        self.assertEqual(len(pt.particle_positions), len(expected_positions))

        for index, position in enumerate(pt.particle_positions):
            np.testing.assert_array_equal(expected_positions[index], position)


class AssociationAndCostMatrixTester(unittest.TestCase):

    def test_the_initialisation_of_the_association_and_cost_matrix(self):
        """
        Test that the initialised association matrix has the correct shape.
        """
        automatic_update = False

        frames_example = np.array([
            [0, 0.1, 0.5],
            [0, 0.6, 0.2],
            [1, 0.1, 0.1],
            [0.1, 0.1, 0.2],
        ], dtype=np.float32)

        times_example = np.array([0, 1, 2, 3])

        particle_positions = [
            np.array([], dtype=np.float32),
            np.array([1], dtype=np.float32),
            np.array([0, 2], dtype=np.float32),
            np.array([0, 2, 3], dtype=np.float32),
        ]

        expected_association_matrix = [
            [
                np.zeros((1, 2), dtype=bool),
                np.zeros((1, 3), dtype=bool),
                np.zeros((1, 4), dtype=bool)
            ],
            [
                np.zeros((2, 3), dtype=bool),
                np.zeros((2, 4), dtype=bool)
            ],
            [
                np.zeros((3, 4), dtype=bool)
            ],
            []
        ]

        expected_cost_matrix = [
            [
                np.zeros((1, 2), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
                np.zeros((1, 4), dtype=np.float32)
            ],
            [
                np.zeros((2, 3), dtype=np.float32),
                np.zeros((2, 4), dtype=np.float32)
            ],
            [
                np.zeros((3, 4), dtype=np.float32)
            ],
            []
        ]

        pt = ParticleTracker(time=times_example, frames=frames_example, automatic_update=automatic_update)
        pt.maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = 2
        pt._particle_positions = particle_positions

        pt._initialise_association_and_cost_matrices()

        self.assertEqual(len(expected_association_matrix), len(pt._association_matrix))
        self.assertEqual(len(expected_cost_matrix), len(pt._cost_matrix))

        for index in range(len(expected_association_matrix)):
            self.assertEqual(len(expected_association_matrix[index]), len(pt._association_matrix[index]), msg=index)
            self.assertEqual(len(expected_cost_matrix[index]), len(pt._cost_matrix[index]), msg=index)

        for frame_index in range(len(expected_association_matrix)):
            for r in range(len(expected_association_matrix[frame_index])):
                np.testing.assert_array_equal(pt._association_matrix[frame_index][r], expected_association_matrix[frame_index][r])
                np.testing.assert_array_equal(pt._cost_matrix[frame_index][r], expected_cost_matrix[frame_index][r])

    def test_calculation_of_zeroth_order_intensity_moments(self):
        """
        Verification of the calulation of first order intensity moments
        TODO: Rewrite to have more logical frames with particle positions going from left to right.
        """

        frames_example = np.array([
            [0.3, 0.1, 0, 0.1, 0.1, 0.5],
            [0, 0.6, 0.2, 0.3, 0.1, 0],
            [1, 0.1, 0.1, 0.3, 0.2, 0],
            [0.1, 0.1, 0.2, 0.2, 0.1, 0.1],
            [0.1, 0.1, 0.8, 0.2, 0.3, 0],
            [0.7, 0.2, 0.2, 0.2, 0.6, 0.1],
            [0.1, 0.8, 0.2, 0.2, 0.9, 0.1]
        ], dtype=np.float32)

        times_example = np.array([0, 1, 2, 3, 4, 5, 6])

        particle_positions = np.array([
            [5, 0],
            [1.25, 1],
            [0, 2],
            [2, 4],
            [0, 5],
            [4, 5],
            [1, 6],
            [4, 6]
        ], dtype=np.float32)

        expected_zeroth_order_intensity_moment_integration_radius_zero = [
            0.5,
            0.6,
            1,
            0.8,
            0.7,
            0.6,
            0.8,
            0.9
        ]

        expected_zeroth_order_intensity_moment_integration_radius_one = [
            0.7,
            0.8,
            1.2,
            1.1,
            1.1,
            0.9,
            1.1,
            1.2
        ]

        expected_zeroth_order_intensity_moment_integration_radius_two = [
            0.9,
            1.4,
            1.4,
            1.5,
            1.5,
            1.3,
            1.5,
            1.6
        ]

        expected_zeroth_order_intensity_moment_integration_radius_three = [
            3.3,
            3.2
        ]

        expected_second_order_intensity_moment_integration_radius_zero = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ]

        expected_second_order_intensity_moment_integration_radius_one = [
            0.2857142857,
            0.25,
            0.1666666667,
            0.2727272727,
            0.3636363636,
            0.3333333333,
            0.2727272727,
            0.25
        ]

        expected_second_order_intensity_moment_integration_radius_two = [
            1.1111111111,
            1.8571428571,
            0.7142857143,
            1.2666666667,
            1.3333333333,
            1.4615384615,
            1.2666666667,
            1.1875
        ]

        expected_second_order_intensity_moment_integration_radius_three = [
            5.48484848485,
            5.09375
        ]

        pt = ParticleTracker(frames=frames_example, time=times_example, automatic_update=False)
        pt.integration_radius_of_intensity_peaks = 0

        for index, position in enumerate(particle_positions):
            np.testing.assert_almost_equal(pt._calculate_zeroth_order_intensity_moment(position[0], int(round(position[1]))),
                                           expected_zeroth_order_intensity_moment_integration_radius_zero[index])

        for index, position in enumerate(particle_positions):
            np.testing.assert_almost_equal(pt._calculate_second_order_intensity_moment(position[0], int(round(position[1]))),
                                           expected_second_order_intensity_moment_integration_radius_zero[index])

        pt.integration_radius_of_intensity_peaks = 1

        for index, position in enumerate(particle_positions):
            np.testing.assert_almost_equal(expected_zeroth_order_intensity_moment_integration_radius_one[index],
                                           pt._calculate_zeroth_order_intensity_moment(position[0], int(round(position[1]))))

        for index, position in enumerate(particle_positions):
            np.testing.assert_almost_equal(pt._calculate_second_order_intensity_moment(position[0], int(round(position[1]))),
                                           expected_second_order_intensity_moment_integration_radius_one[index])

        pt.integration_radius_of_intensity_peaks = 2

        for index, position in enumerate(particle_positions):
            np.testing.assert_almost_equal(pt._calculate_zeroth_order_intensity_moment(position[0], int(round(position[1]))),
                                           expected_zeroth_order_intensity_moment_integration_radius_two[index], decimal=5)

        for index, position in enumerate(particle_positions):
            np.testing.assert_almost_equal(pt._calculate_second_order_intensity_moment(position[0], int(round(position[1]))),
                                           expected_second_order_intensity_moment_integration_radius_two[index])

        pt.integration_radius_of_intensity_peaks = 3

        np.testing.assert_almost_equal(pt._calculate_zeroth_order_intensity_moment(1, 6),
                                       expected_zeroth_order_intensity_moment_integration_radius_three[0])

        np.testing.assert_almost_equal(pt._calculate_zeroth_order_intensity_moment(4, 6),
                                       expected_zeroth_order_intensity_moment_integration_radius_three[1])

        np.testing.assert_almost_equal(pt._calculate_second_order_intensity_moment(1, 6),
                                       expected_second_order_intensity_moment_integration_radius_three[0], decimal=5)

        np.testing.assert_almost_equal(pt._calculate_second_order_intensity_moment(4, 6),
                                       expected_second_order_intensity_moment_integration_radius_three[1])

    def test_calculation_of_the_cost_matrix(self):
        """
        Test to verify that the calculation of the cost matrix is done correctly.
        """
        frames_example = np.array([
            [0, 0.1, 0.5],
            [0, 0.6, 0.2],
            [1, 0.1, 0.1],
            [0.1, 0.1, 0.2],
            [0.7, 0.2, 0.7]
        ], dtype=np.float32)

        times_example = np.array([0, 1, 2, 3, 4])

        particle_positions = [
            np.array([2], dtype=np.float32),
            np.array([1.25], dtype=np.float32),
            np.array([0], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([0, 2], dtype=np.float32)
        ]

        max_frames = 2
        max_distance = 1

        expected_cost_matrix = [
            [
                np.array(
                    [
                        [0, 1],
                        [1, 0.5737755],
                    ], dtype=np.float32),
                np.array(
                    [
                        [0, 4],
                        [4, np.inf]
                    ], dtype=np.float32),
                np.array(
                    [
                        [0],
                        [9]
                    ], dtype=np.float32)
            ],
            [
                np.array(
                    [
                        [0, 1],
                        [1, np.inf]
                    ], dtype=np.float32),
                np.array(
                    [
                        [0],
                        [4]
                    ], dtype=np.float32),
                np.array(
                    [
                        [0, 9, 9],
                        [9, 1.6654132, 0.6654132]
                    ], dtype=np.float32)

            ],
            [
                np.array(
                    [
                        [0],
                        [1]
                    ], dtype=np.float32),
                np.array(
                    [
                        [0, 4, 4],
                        [4, 0.0487971, np.inf]
                    ], dtype=np.float32)

            ],
            [
                np.array(
                    [
                        [0, 1, 1],
                    ], dtype=np.float32)
            ],
            []
        ]

        pt = ParticleTracker(frames=frames_example, time=times_example, automatic_update=False)
        pt.integration_radius_of_intensity_peaks = 1
        pt.maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = 2
        pt.maximum_distance_a_particle_can_travel_between_frames = 1

        pt._particle_positions = particle_positions
        pt._calculate_particle_moments()

        pt._initialise_association_and_cost_matrices()
        pt._calculate_cost_matrices()

        for frame_index, _ in enumerate(pt._cost_matrix):
            for future_frame_index, _ in enumerate(pt._cost_matrix[frame_index]):
                np.testing.assert_array_almost_equal(pt._cost_matrix[frame_index][future_frame_index], expected_cost_matrix[frame_index][future_frame_index])

    def test_initial_links_in_association_matrix(self):
        """
        Verification of the initial links in the association matrix is set correctly.
        """
        frames_example = np.array([
            [0, 0.1, 0.5],
            [0, 0.6, 0.2],
            [1, 0.1, 0.1],
        ], dtype=np.float32)

        times_example = np.array([0, 1, 2])

        cost_matrix = [
            [
                np.array(
                    [
                        [0, 1, 1, 1],
                        [1, 0.2, 0.5, 0.3],
                        [1, 0.1, 0.6, 0.3],
                        [1, 0.2, 0.3, np.inf],
                        [1, 0.3, 0.1, 0.8]
                    ], dtype=np.float32),
                np.array(
                    [
                        [0, 4, 4],
                        [4, 5, 6],
                        [4, 7, 8],
                        [4, 9, 10]
                    ], dtype=np.float32)
            ]
        ]
        initial_association_matrix = [
            [
                np.zeros((5, 4), dtype=bool),
                np.zeros((4, 3), dtype=bool)
            ]
        ]

        expected_association_matrix = [
            [
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0]
                    ], dtype=bool),
                np.array(
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0]
                    ], dtype=bool)

            ]
        ]

        pt = ParticleTracker(frames=frames_example, time=times_example, automatic_update=False)

        pt._cost_matrix = cost_matrix
        pt._association_matrix = initial_association_matrix
        pt._create_initial_links_in_association_matrix()

        for frame_index, _ in enumerate(pt._cost_matrix):
            for future_frame_index, _ in enumerate(pt._cost_matrix[frame_index]):
                np.testing.assert_array_almost_equal(pt._association_matrix[frame_index][future_frame_index], expected_association_matrix[frame_index][future_frame_index])

    def test_optimisation_of_association_matrix(self):
        """
        Test that the optimisation of the association matrix minimises the cost.
        """
        frames_example = np.array([
            [0, 0.1, 0.5],
            [0, 0.6, 0.2],
            [1, 0.1, 0.1],
        ], dtype=np.float32)

        times_example = np.array([0, 1, 2])

        cost_matrix = [
            [
                np.array(
                    [
                        [0, 1, 1],
                        [1, 0.8, 0.9],
                        [1, 0.1, 0.9],
                    ], dtype=np.float32),
                np.array(
                    [
                        [0, 4, 4],
                        [4, 11, 12],
                        [4, 14, 13],
                        [4, 3, 14]
                    ], dtype=np.float32),
                np.array(
                    [
                        [0, 4, 4, 4],
                        [4, 13, 14, 3],
                        [4, 14, 5, 1]
                    ], dtype=np.float32)
            ]
        ]

        initial_association_matrix = [
            [
                np.zeros((3, 3), dtype=bool),
                np.zeros((4, 3), dtype=bool),
                np.zeros((3, 4), dtype=bool)
            ]
        ]

        expected_association_matrix = [
            [
                np.array(
                    [
                        [1, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                    ], dtype=bool),
                np.array(
                    [
                        [1, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                    ], dtype=bool),
                np.array(
                    [
                        [1, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]
                    ], dtype=bool)
            ]
        ]

        pt = ParticleTracker(frames=frames_example, time=times_example, automatic_update=False)

        pt._cost_matrix = cost_matrix
        pt._association_matrix = initial_association_matrix
        pt._create_initial_links_in_association_matrix()
        pt._optimise_association_matrix()

        for frame_index, _ in enumerate(pt._association_matrix):
            for future_frame_index, _ in enumerate(pt._association_matrix[frame_index]):
                np.testing.assert_array_almost_equal(pt._association_matrix[frame_index][future_frame_index], expected_association_matrix[frame_index][future_frame_index])

    def test_creating_trajectory(self):

        frames_example = np.array([
            [0, 0.1, 0.5],
            [0, 0.6, 0.2],
            [1, 0.1, 0.1],
            [1, 0.1, 0.1]
        ], dtype=np.float32)

        times_example = np.array([1, 2, 3, 4])

        particle_position = [
            np.array([0, 1]),
            np.array([0, 1, 2]),
            np.array([0, 1]),
            np.array([0, 1, 2]),
        ]

        zeroth_order_moments = [
            np.array([1, 2]),
            np.array([1, 2, 3]),
            np.array([1, 2]),
            np.array([1, 2, 3]),
        ]

        second_order_moments = [
            np.array([2, 3]),
            np.array([2, 3, 4]),
            np.array([2, 3]),
            np.array([2, 3, 4]),
        ]

        association_matrix = [
            [

                np.array(
                    [
                        [1, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                    ], dtype=bool),
                np.array(
                    [
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 0, 1],
                    ], dtype=bool),
                np.array(
                    [
                        [1, 0, 0, 1],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]
                    ], dtype=bool)
            ],
            [
                np.array(
                    [
                        [1, 1, 1],
                        [0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0]
                    ], dtype=bool),
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ], dtype=bool)
            ],
            [
                np.array(
                    [
                        [1, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0]
                    ], dtype=bool),
            ],
            []
        ]

        trajectories = [
            np.zeros((4,),
                     dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32), ('zeroth_order_moment', np.float32), ('second_order_moment', np.float32)]),
            np.zeros((4,),
                     dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32), ('zeroth_order_moment', np.float32), ('second_order_moment', np.float32)]),
            np.zeros((2,),
                     dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32), ('zeroth_order_moment', np.float32), ('second_order_moment', np.float32)])
        ]

        trajectories[0]['frame_index'] = np.array([0, 1, 2, 3])
        trajectories[0]['time'] = np.array([1, 2, 3, 4])
        trajectories[0]['position'] = np.array([0, 1, 0, 1])
        trajectories[0]['zeroth_order_moment'] = np.array([1, 2, 1, 2])
        trajectories[0]['second_order_moment'] = np.array([2, 3, 2, 3])

        trajectories[1]['frame_index'] = np.array([0, 1, 2, 3])
        trajectories[1]['time'] = np.array([1, 2, 3, 4])
        trajectories[1]['position'] = np.array([1, 0, 1, 0])
        trajectories[1]['zeroth_order_moment'] = np.array([2, 1, 2, 1])
        trajectories[1]['second_order_moment'] = np.array([3, 2, 3, 2])

        trajectories[2]['frame_index'] = np.array([1, 3])
        trajectories[2]['time'] = np.array([2, 4])
        trajectories[2]['position'] = np.array([2, 2])
        trajectories[2]['zeroth_order_moment'] = np.array([3, 3])
        trajectories[2]['second_order_moment'] = np.array([4, 4])

        pt = ParticleTracker(time=times_example, frames=frames_example, automatic_update=False)

        pt._association_matrix = association_matrix
        pt._particle_positions = particle_position
        pt._zeroth_order_moments = zeroth_order_moments
        pt._second_order_moments = second_order_moments

        pt._update_trajectories()

        for index, t in enumerate(pt.trajectories):
            np.testing.assert_array_equal(trajectories[index], t.particle_positions)


if __name__ == '__main__':
    unittest.main()
