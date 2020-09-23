import unittest
import numpy as np
from particle_tracker_one_d import ShortestPathFinder
import pprint


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
            np.array([0, 1, 0, 0.1], dtype=np.float32),
            np.array([[0.2], [0.4]], dtype=np.float32),
        ]
        non_valid_type_of_frames = [
            2,
            [[0, 1, 0], [0, 0, 0]],
            'string'
        ]
        for frames in valid_frames:
            self.assertTrue(ShortestPathFinder._test_if_frames_have_correct_format(frames), msg='Valid frames not accepted, frames: ' + np.array_str(frames))

        for frames in non_valid_shape_or_value_frames:
            with self.assertRaises(ValueError):
                ShortestPathFinder._test_if_frames_have_correct_format(frames)

        for frames in non_valid_type_of_frames:
            with self.assertRaises(TypeError):
                ShortestPathFinder._test_if_frames_have_correct_format(frames)

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
            self.assertTrue(ShortestPathFinder._test_if_time_has_correct_format(times), msg='Valid times not accepted, times: ' + np.array_str(times))

        for times in non_valid_shape_or_values_times:
            with self.assertRaises(ValueError):
                ShortestPathFinder._test_if_time_has_correct_format(times)

        for times in non_valid_types_of_times:
            with self.assertRaises(TypeError):
                ShortestPathFinder._test_if_time_has_correct_format(times)

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
            self.assertTrue(ShortestPathFinder._test_if_time_and_frames_has_same_length(time_and_frames['time'], time_and_frames['frames']))

        for time_and_frames in non_valid_times_and_frames:
            with self.assertRaises(ValueError):
                ShortestPathFinder._test_if_time_and_frames_has_same_length(time_and_frames['time'], time_and_frames['frames'])

    def test_validation_of_automatic_update(self):

        valid_automatic_update_arguments = [True, False]

        non_valid_automatic_update_arguments = [0, 1, 'True', 'False', np.array, []]

        for automatic_update in valid_automatic_update_arguments:
            self.assertTrue(ShortestPathFinder._test_if_automatic_update_has_correct_format(automatic_update))

        for automatic_update in non_valid_automatic_update_arguments:
            with self.assertRaises(ValueError):
                ShortestPathFinder._test_if_automatic_update_has_correct_format(automatic_update)

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

        pt = ShortestPathFinder(frames=frames, time=time, automatic_update=automatic_update)

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

        pt = ShortestPathFinder(frames=frames, time=time, automatic_update=automatic_update)

        for width in valid_boxcar_widths:
            pt.boxcar_width = width
            self.assertEqual(pt.boxcar_width, width)

        for width in non_valid_type_of_boxcar_widths:
            with self.assertRaises(TypeError, msg=width):
                pt.boxcar_width = width

        for width in non_valid_values_of_boxcar_widths:
            with self.assertRaises(ValueError, msg=width):
                pt.boxcar_width = width

    def test_validation_of_setting_start_point(self):
        """
        Tests the setting of the class attribute start_point. Should be a tuple, list or np.array
        """

        frames = np.array([
            [0, 0.1, 0.2, 0.1],
            [0, 0.2, 0.3, 0.4],
            [0.2, 0.5, 0.6, 1],
            [0, 0.1, 0.2, 0.1]
        ], dtype=np.float32)
        time = np.array([0, 1, 2, 3])
        automatic_update = False

        valid_start_points = [
            (0, 0),
            [0, 1],
            np.array([1, 1], dtype=np.int16),
            np.array([1, 3], dtype=np.int32),
        ]
        non_valid_type_of_start_points = [
            1.5,
            '12',
            None,
            (0.3, 2),
            ['', 4],
            np.array([0, 2], dtype=np.float32)
        ]
        non_valid_values_of_start_points = [
            (0, -1),
            (4, 1),
            (0, 10)
        ]

        spf = ShortestPathFinder(frames=frames, time=time, automatic_update=automatic_update)

        for start_point in valid_start_points:
            spf.start_point = start_point
            self.assertEqual(spf.start_point, (start_point[0], start_point[1]))

        for start_point in non_valid_type_of_start_points:
            with self.assertRaises(TypeError, msg=start_point):
                spf.start_point = start_point

        for start_point in non_valid_values_of_start_points:
            with self.assertRaises(ValueError, msg=start_point):
                spf.start_point = start_point

    def test_validation_of_setting_end_point(self):
        """
        Tests the setting of the class attribute end_point. Should be a tuple, list or np.array
        """

        frames = np.array([
            [0, 0.1, 0.2, 0.1],
            [0, 0.2, 0.3, 0.4],
            [0.2, 0.5, 0.6, 1],
            [0, 0.1, 0.2, 0.1]
        ], dtype=np.float32)
        time = np.array([0, 1, 2, 3])
        automatic_update = False

        valid_end_points = [
            (0, 0),
            [0, 1],
            np.array([1, 1], dtype=np.int16),
            np.array([1, 3], dtype=np.int32),
            np.array([3, 3], dtype=np.int32)
        ]
        non_valid_type_of_end_points = [
            1.5,
            '12',
            None,
            (0.3, 2),
            ['', 4],
            np.array([0, 2], dtype=np.float32)
        ]
        non_valid_values_of_end_points = [
            (0, -1),
            (4, 1),
            (0, 10)
        ]

        spf = ShortestPathFinder(frames=frames, time=time, automatic_update=automatic_update)

        for end_point in valid_end_points:
            spf.end_point = end_point
            self.assertEqual(spf.end_point, (end_point[0], end_point[1]))

        for end_point in non_valid_type_of_end_points:
            with self.assertRaises(TypeError, msg=end_point):
                spf.end_point = end_point

        for end_point in non_valid_values_of_end_points:
            with self.assertRaises(ValueError, msg=end_point):
                spf.end_point = end_point


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

        spf = ShortestPathFinder(frames=frames_examples, time=times_examples, automatic_update=False)
        spf.particle_detection_threshold = 0.1

        for index, intensity in enumerate(intensity_examples):
            np.testing.assert_array_equal(expected_positions[index], spf._find_local_maximas_larger_than_threshold(intensity))

    def test_finding_initial_particle_positions(self):
        """
        Test finding the initial particle positions, that is local maximas between start point and end point
        """

        frames = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 1, 1],
        ], dtype=np.float32)

        time = np.arange(frames.shape[0])
        automatic_update = False

        start_point = (0, 1)
        end_point = (5, 3)

        expected_positions = [
            np.array([1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            np.array([2], dtype=np.float32),
            np.array([3], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([3], dtype=np.float32),
        ]

        spf = ShortestPathFinder(frames, time, automatic_update)
        spf.start_point = start_point
        spf.end_point = end_point

        spf._find_initial_particle_positions()

        for index, positions in enumerate(spf._particle_positions):
            np.testing.assert_array_equal(expected_positions[index], positions)

    def test_the_refining_of_particle_positions(self):
        """
        Test that the refinement of particle positions work. That is, by finding the center of mass close to the initial particle position.
        """
        automatic_update = False

        frames = np.array([
            [0, 0.1, 0.5, 0],
            [0, 0.6, 0.2, 0],
            [1, 0.1, 0.1, 0],
            [0.1, 0.1, 0.2, 0],
            [0, 0.7, 0.2, 0.7]
        ], dtype=np.float32)

        time = np.array([0, 1, 2, 3, 4])

        expected_positions = [
            np.array([1.8333333], dtype=np.float32),
            np.array([1.25], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([1.6666666], dtype=np.float32),
            np.array([2], dtype=np.float32)
        ]

        spf = ShortestPathFinder(time=time, frames=frames, automatic_update=automatic_update)
        spf.start_point = (0, 2)
        spf.end_point = (4, 2)
        spf.integration_radius_of_intensity_peaks = 1
        spf._find_initial_particle_positions()
        spf._refine_particle_positions()

        self.assertEqual(len(spf.particle_positions), len(expected_positions))

        for index, position in enumerate(spf.particle_positions):
            np.testing.assert_array_equal(expected_positions[index], position)


class AssociationAndCostMatrixTester(unittest.TestCase):

    def test_the_initialisation_of_the_association_and_cost_matrix(self):
        """
        Test that the initialised association matrix has the correct shape.
        """
        automatic_update = False

        frames = np.array([
            [0, 0.1, 0.5],
            [0, 0.6, 0.2],
            [1, 0.1, 0.1],
            [0.1, 0.1, 0.2],
            [0.1, 0.1, 0.2],
        ], dtype=np.float32)

        times = np.array([0, 1, 2, 3, 4])

        particle_positions = [
            np.array([0], dtype=np.float32),
            np.array([1], dtype=np.float32),
            np.array([0, 2], dtype=np.float32),
            np.array([0, 2, 3], dtype=np.float32),
            np.array([4], dtype=np.float32),
        ]

        expected_association_matrix = [
            np.zeros((1, 1), dtype=bool),
            np.zeros((1, 2), dtype=bool),
            np.zeros((2, 3), dtype=bool),
            np.zeros((3, 1), dtype=bool)
        ]

        expected_cost_matrix = [
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((3, 1), dtype=np.float32)
        ]

        spf = ShortestPathFinder(time=times, frames=frames, automatic_update=automatic_update)
        spf._particle_positions = particle_positions
        spf._initialise_association_and_cost_matrix()

        association_matrix, cost_matrix = spf._association_matrix, spf._cost_matrix

        self.assertEqual(len(expected_association_matrix), len(association_matrix))
        self.assertEqual(len(expected_cost_matrix), len(cost_matrix))

        for frame_index in range(len(association_matrix)):
            np.testing.assert_array_equal(association_matrix[frame_index], expected_association_matrix[frame_index])
            np.testing.assert_array_equal(cost_matrix[frame_index], expected_cost_matrix[frame_index])

    def test_calculation_of_the_cost_matrix(self):
        """
        Test to verify that the calculation of the cost matrix is done correctly.
        """
        frames = np.array([
            [0, 0.3, 0.5],
            [0, 0.1, 0.5],
            [0, 0.6, 0.2],
            [1, 0.1, 0.1],
            [0.7, 0.2, 0.7],
            [0.1, 0.1, 0.2],
            [0.1, 0.1, 0.2]
        ], dtype=np.float32)

        times = np.array([0, 1, 2, 3, 4, 5, 6])

        particle_positions = [
            np.array([2], dtype=np.float32),
            np.array([1.25], dtype=np.float32),
            np.array([0], dtype=np.float32),
            np.array([0, 2], dtype=np.float32),
            np.array([2], dtype=np.float32)
        ]

        start_point = (1, 2)
        end_point = (5, 2)

        expected_cost_matrix = [
            np.array(
                [
                    [0.06625],
                ], dtype=np.float32),
            np.array(
                [
                    [0.4725],
                ], dtype=np.float32),
            np.array(
                [
                    [0.6325, 1.0325],
                ], dtype=np.float32),
            np.array(
                [
                    [1.1225],
                    [1.1225001],
                ], dtype=np.float32),
        ]

        spf = ShortestPathFinder(frames=frames, time=times, automatic_update=False)
        spf.start_point = start_point
        spf.end_point = end_point

        spf._particle_positions = particle_positions

        spf._initialise_association_and_cost_matrix()
        spf._calculate_particle_moments()
        spf._calculate_cost_matrix()

        print(expected_cost_matrix)
        print(spf._cost_matrix)

        for frame_index, _ in enumerate(spf._cost_matrix):
            for future_frame_index, _ in enumerate(spf._cost_matrix[frame_index]):
                np.testing.assert_array_almost_equal(spf._cost_matrix[frame_index][future_frame_index], expected_cost_matrix[frame_index][future_frame_index])

    def test_find_shortest_path_by_dijkstra(self):
        """
         Test automatic finding
         """

        automatic_update = True

        frames = np.array([
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0]
        ], dtype=np.float32)

        times = np.array([0, 1, 2, 3, 4])

        start_point = (0, 1)
        end_point = (4, 4)

        spf = ShortestPathFinder(frames=frames, time=times, automatic_update=automatic_update)
        spf.boxcar_width = 0
        spf.start_point = start_point
        spf.end_point = end_point

        #print('1.')
        pprint.pprint(spf._create_trajectory_from_cost_matrix())
        #print('2.')
