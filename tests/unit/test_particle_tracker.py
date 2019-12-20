import unittest
import numpy as np
from particle_tracker_one_d import ParticleTracker


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
                ParticleTracker._test_if_time_and_frames_has_same_length(time_and_frames['time'],time_and_frames['frames'])

    def test_set_boxcar_width(self):
        """
        Test that the boxcar_width is an integer larger than 0 but smaller than the half the number of pixels in a frame.
        """
        frames = np.zeros(1)

    def test_set_integration_radius(self):
        """

        :return:
        """
        valid_values = [1, 10, 13]


if __name__ == '__main__':
    unittest.main()
