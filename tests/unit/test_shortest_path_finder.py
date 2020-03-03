import unittest
import numpy as np
from particle_tracker_one_d import Sho

class SetAttributeTester(unittest.TestCase):
    mock_frames = np.zeros((2, 3), dtype=np.float32)

    def test_validation_of_frames_argument(self):
        """
        Test that the frames are a numpy array with shape (nFrames,nPixels) with float values between 0 and 1.0.
        The number of frames nFrames should be larger than one and the nPixels should be larger than 2.
        The frames should contain at least one value equal to 1 and one value equal to 0.
        """
        valid_frames = [
            np.array([[0, 1, 0], [0, 0, 0]], dtype=np.float32),
            np.array([[0.1, 0, 0.34], [0.1, 0, 1], [0.12, 0.003, 0.9]], dtype=np.float32),
        ]
        non_valid_shape_or_value_frames = [
            np.array([[0, 1, 0]], dtype=np.float32),
            np.array([[0, 1], [0, 0.1]], dtype=np.float32),
            np.array([[0, 2, 0], [0, 0, 0]], dtype=np.float32),
            np.array([[0.2, 1, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
            np.array([[[0.1, 2, 3], [0.1, 0.3, 1]]], dtype=np.float32),
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
