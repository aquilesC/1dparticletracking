import unittest
import numpy as np
from particle_tracker_one_d import Frames


class ValidationOfArgumentsTester(unittest.TestCase):

    def test_validation_of_frames(self):
        """
        Test that the frames are a numpy array with shape (nFrames,nPixels).
        """

        valid_frames = [
            np.random.rand(1, 2),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 21), dtype=np.int8),
            np.zeros((13, 2), dtype=np.int16),
            np.zeros((1, 20), dtype=np.float64),
            np.random.rand(2, 1),
            np.random.rand(20, 100)
        ]

        non_valid_shape_frames = [
            np.random.rand(1, 1, 1),
            np.random.rand(0, 1),
            np.random.rand(1, 0),
            np.random.rand(1, 0, 3, 4),
        ]

        non_valid_type_of_frames = [
            2,
            [[0, 1, 0], [0, 0, 0]],
            'string'
        ]

        for frames in valid_frames:
            self.assertTrue(Frames._test_if_frames_have_correct_format(frames), msg='Valid frames not accepted, frames: ' + np.array_str(frames))

        for frames in non_valid_shape_frames:
            with self.assertRaises(ValueError):
                Frames._test_if_frames_have_correct_format(frames)

        for frames in non_valid_type_of_frames:
            with self.assertRaises(TypeError):
                Frames._test_if_frames_have_correct_format(frames)

    def test_validation_time_argument(self):
        """
        Test that the validation of the class argument works. The time should be a numpy.ndarray with monotonically increasing values.
        """
        valid_times = [
            np.array([0, 1, 2, 4, 6], dtype=np.float32),
            np.array([0, 1000], dtype=np.int16),
            np.array([0.1, 0.2, 0.4], dtype=np.float32),
            np.array([1], dtype=np.float32)
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
            self.assertTrue(Frames._test_if_time_has_correct_format(times), msg='Valid times not accepted, times: ' + np.array_str(times))

        for times in non_valid_shape_or_values_times:
            with self.assertRaises(ValueError):
                Frames._test_if_time_has_correct_format(times)

        for times in non_valid_types_of_times:
            with self.assertRaises(TypeError):
                Frames._test_if_time_has_correct_format(times)


class ImageRestorationTester(unittest.TestCase):

    def test_boxcar_averaging(self):
        """
        Test that the boxcar average works as it should.
        """

        frames = np.array([
            [1, 1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1],
            [0, 1, 2, 3, 4, 5]
        ], dtype=np.float32)

        time = np.array([0, 1, 2, 3], dtype=np.float32)

        boxcar_widths = [0, 1, 2, 3]

        expected_averaged_frames = [
            np.array([
                [1, 1, 1, 1, 1, 1],
                [0, 1, 0, 1, 0, 1],
                [0, 0, 1, 0, 0, 1],
                [0, 1, 2, 3, 4, 5]
            ], dtype=np.float32),
            np.array([
                [1, 1, 1, 1, 1, 1],
                [0, 1, 0, 1, 0, 1],
                [0, 0, 1, 0, 0, 1],
                [0, 1, 2, 3, 4, 5]
            ], dtype=np.float32),
            np.array([
                [1, 1, 1, 1, 1],
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0, 0.5, 0.5, 0, 0.5],
                [0.5, 1.5, 2.5, 3.5, 4.5]
            ], dtype=np.float32),
            np.array([
                [1, 1, 1, 1],
                [0.33333333, 0.666666667, 0.33333333, 0.66666667],
                [0.33333333, 0.33333333, 0.33333333, 0.33333333],
                [1, 2, 3, 4]
            ], dtype=np.float32)
        ]

        frames = Frames(frames=frames, time=time, automatic_update=True)

        for index, boxcar_width in enumerate(boxcar_widths):
            frames.boxcar_width = boxcar_width
            np.testing.assert_array_equal(frames.frames, expected_averaged_frames[index])
