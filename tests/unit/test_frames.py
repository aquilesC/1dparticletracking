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
