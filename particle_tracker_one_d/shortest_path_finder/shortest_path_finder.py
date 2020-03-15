from ..particle_tracker import ParticleTracker
import numpy as np
from astropy.convolution import convolve, Box1DKernel
import matplotlib.pyplot as plt
from ..trajectory import Trajectory


class ShortestPathFinder:
    """
    Class for finding shortest path between points in a set of frames.

    Parameters
    ----------
    frames: np.array
        The frames in which trajectories are to be found. The shape of the np.array should be (nFrames,xPixels). The intensity of the frames should be normalised according to
        :math:`I_n = (I-I_{min})/(I_{max}-I_{min})`, where :math:`I` is the intensity of the frames, :math:`I_{min}`, :math:`I_{max}` are the global intensity minima and maxima of the
        frames.
    time: np.array
        The corresponding time of each frame.
    automatic_update: bool
        Choose if the class should update itself when changing properties.

    Attributes
    ----------
    frames
    time
    boxcar_width
    integration_radius_of_intensity_peaks
    particle_detection_threshold
    particle_positions
    """

    def __init__(self, frames, time, automatic_update=True):
        ParticleTracker._validate_class_arguments(frames, time, automatic_update)
        self._automatic_update = automatic_update
        self._frames = frames
        self._time = time
        self._integration_radius_of_intensity_peaks = 1
        self._boxcar_width = 0
        self._averaged_intensity = frames
        self._particle_positions = [None] * self.frames.shape[0]
        self._cost_matrix = []
        self._association_matrix = []
        self._start_point = (0, 0)
        self._end_point = (0, 0)
        self._shortest_path = None
        self._trajectory = None

    @property
    def frames(self):
        """
        np.array:
            The frames which the particle tracker tries to find trajectories in. If the property boxcar_width!=0 it will return the smoothed frames.
        """
        return self._averaged_intensity

    @property
    def boxcar_width(self):
        """
        int:
            Number of values used in the boxcar averaging of the frames.
        """
        return self._boxcar_width

    @boxcar_width.setter
    def boxcar_width(self, width):
        if type(width) is not int:
            raise TypeError('Attribute boxcar_width should be of type int')
        if not -1 < width <= self.frames.shape[1]:
            raise ValueError('Attribute boxcar_width should be a positive integer less or equal the number of pixels in each frame.')

        if not width == self._boxcar_width:
            self._boxcar_width = width
            if self._automatic_update:
                self._update_averaged_intensity()
                self._find_particle_positions()
                self._update_shortest_path()

    @property
    def integration_radius_of_intensity_peaks(self):
        """
        int:
            Number of pixels used when integrating the intensity peaks. No particles closer than twice this value will be found. If two peaks are found within twice this value,
            the one with highest intensity moment will be kept.
        """
        return self._integration_radius_of_intensity_peaks

    @integration_radius_of_intensity_peaks.setter
    def integration_radius_of_intensity_peaks(self, radius):
        if type(radius) is not int:
            raise TypeError('Attribute integration_radius_of_intensity_peaks should be of type int')
        if not -1 < radius <= self.frames.shape[1] / 2:
            raise ValueError('Attribute integration_radius_of_intensity_peaks should be a positive integer less or equal the half of the number of pixels in each frame.')

        if not radius == self._integration_radius_of_intensity_peaks:
            self._integration_radius_of_intensity_peaks = radius
            if self._automatic_update:
                self._find_particle_positions()
                self._update_shortest_path()

    @property
    def start_point(self):
        """
        tuple:
            (frame_index, position_index), The start point of the path you want to find.
        """
        return self._start_point

    @start_point.setter
    def start_point(self, start_point):
        if type(start_point) not in [list, tuple, np.ndarray]:
            raise TypeError('Start point must be list, tuple or np.array with form (frame_index, position_index)')
        else:
            for val in start_point:
                if type(val) not in [int, np.int16, np.int32]:
                    raise TypeError('Start point values must be integers')
            if start_point[0] < 0 or start_point[0] > self._time.shape[0] - 1:
                raise ValueError('First value in start point must be an integer between 0 and self.time.shape[0] - 1')
            elif start_point[1] < 0 or start_point[1] > self._frames.shape[1] - 1:
                raise ValueError('Second value in start point must be an integer between 0 and self.frames.shape[1] - 1')

            if not (start_point[0] == self._start_point[0] and start_point[1] == self._start_point[1]):
                self._start_point = (int(start_point[0]), int(start_point[1]))
                if self._automatic_update:
                    self._find_particle_positions()
                    self._update_shortest_path()

    @property
    def end_point(self):
        """
        tuple:
            (frame_index, position_index), The end point of the path you want to find.
        """
        return self._end_point

    @end_point.setter
    def end_point(self, end_point):
        if type(end_point) not in [list, tuple, np.ndarray]:
            raise TypeError('End point must be list, tuple or np.array with form (frame_index, position_index)')
        else:
            for val in end_point:
                if type(val) not in [int, np.int16, np.int32]:
                    raise TypeError('End point values must be integers')
            if end_point[0] < 0 or end_point[0] > self._time.shape[0] - 1:
                raise ValueError('First value in end point must be an integer between 0 and self.time.shape[0] - 1')
            elif end_point[1] < 0 or end_point[1] > self._frames.shape[1] - 1:
                raise ValueError('Second value in end point must be an integer between 0 and self.frames.shape[1] - 1')

            if not (end_point[0] == self._end_point[0] and end_point[1] == self._end_point[1]):
                self._end_point = (int(end_point[0]), int(end_point[1]))
                if self._automatic_update:
                    self._find_particle_positions()
                    self._update_shortest_path()

    @property
    def particle_positions(self):
        """
        np.array:
            Numpy array with all particle positions on the form `np.array((nParticles,), dtype=[('frame_index', np.int16),
            ('time', np.float32),('integer_position', np.int16), ('refined_position', np.float32)])`
        """
        return self._particle_positions

    @property
    def shortest_path(self):
        """
        dict:
            The shortest path between the start and end point, defined by the cost function. Cost, length and association matrix.
        """
        return self._shortest_path

    @property
    def trajectory(self):
        """
        trajectory:
            The shortest path between the start and end point represented as a trajectory.
        """
        return self._trajectory

    @property
    def _intensity_of_interest(self):
        return self._averaged_intensity[self.start_point[0]:self.end_point[0] + 1]

    @property
    def _time_of_interest(self):
        return self._time[self.start_point[0]:self.end_point[0] + 1]

    def _find_particle_positions(self):
        if self.start_point and self.end_point and self.start_point[0] < self.end_point[0]:
            self._find_initial_particle_positions()
            self._refine_particle_positions()

    def _refine_particle_positions(self):
        if self._integration_radius_of_intensity_peaks == 0:
            return
        for frame_index, positions in enumerate(self._particle_positions):
            for index, position in enumerate(positions):
                if position == 0 or position + 1 == self._intensity_of_interest.shape[1]:
                    continue
                elif position < self.integration_radius_of_intensity_peaks:
                    integration_radius = position
                elif position > self._intensity_of_interest.shape[1] - self._integration_radius_of_intensity_peaks - 1:
                    integration_radius = self._intensity_of_interest.shape[1] - position
                else:
                    integration_radius = self.integration_radius_of_intensity_peaks
                intensity = self._intensity_of_interest[frame_index][int(position - integration_radius):int(position + integration_radius + 1)]
                intensity = intensity - np.min(intensity)
                self._particle_positions[frame_index][index] = position + ParticleTracker._calculate_center_of_mass(intensity) - integration_radius

    def _find_initial_particle_positions(self):
        self._particle_positions = [None] * (self.end_point[0] - self.start_point[0] + 1)
        for index, frame in enumerate(self._intensity_of_interest):
            self._particle_positions[index] = self._find_local_maximas(frame)
        self._particle_positions[0] = np.array([self.start_point[1]], dtype=np.float32)
        self._particle_positions[-1] = np.array([self.end_point[1]], dtype=np.float32)

    def _initialise_association_and_cost_matrix(self):
        number_of_frames = len(self._particle_positions) - 1

        self._association_matrix = [[] for _ in range(number_of_frames)]
        self._cost_matrix = [[] for _ in range(number_of_frames)]

        for frame_index in range(0, number_of_frames):
            self._association_matrix[frame_index] = np.zeros(
                (len(self._particle_positions[frame_index]), len(self._particle_positions[frame_index + 1])),
                dtype=bool
            )
            self._cost_matrix[frame_index] = np.zeros(
                (len(self._particle_positions[frame_index]), len(self._particle_positions[frame_index + 1])),
                dtype=np.float32
            )

    def _calculate_cost_matrix(self):
        for frame_index, _ in enumerate(self._cost_matrix):
            for particle_index, _ in enumerate(self._cost_matrix[frame_index]):
                for future_particle_index, _ in enumerate(self._cost_matrix[frame_index][particle_index]):
                    particle_position_in_current_frame = (frame_index + self.start_point[0], self._particle_positions[frame_index][particle_index])
                    particle_position_in_future_frame = (frame_index + self.start_point[0] + 1, self._particle_positions[frame_index + 1][future_particle_index])
                    if frame_index == 0:
                        self._cost_matrix[frame_index][particle_index][future_particle_index] = self._calculate_linking_cost(
                            particle_position_in_current_frame, particle_position_in_future_frame)
                    else:
                        self._cost_matrix[frame_index][particle_index][future_particle_index] = (
                                self._calculate_linking_cost(particle_position_in_current_frame, particle_position_in_future_frame) +
                                np.amin(self._cost_matrix[frame_index - 1].T[particle_index])
                        )

    def _calculate_linking_cost(self, position1, position2):
        return (
                0.1*np.abs((position1[1] - position2[1])) ** 2 +
                #(self._calculate_first_order_intensity_moment(position1[1], position1[0]) - self._calculate_first_order_intensity_moment(
                #    position2[1], position2[0])) ** 2 +
                #(self._calculate_second_order_intensity_moment(
                #    position1[1], position1[0]) - self._calculate_second_order_intensity_moment(position2[1], position2[0])) ** 2 +
                (self._calculate_first_order_intensity_moment(self.start_point[1], self.start_point[0]) - self._calculate_first_order_intensity_moment(
                    position2[1], position2[0])) ** 2
                #(self._calculate_second_order_intensity_moment(
                #    self.start_point[1], self.start_point[0]) - self._calculate_second_order_intensity_moment(position2[1], position2[0])) ** 2
        )

    def _update_shortest_path(self):
        if self.start_point and self.end_point and self.start_point[0] < self.end_point[0]:
            self._initialise_association_and_cost_matrix()
            self._calculate_cost_matrix()
            self._create_trajectory_from_cost_matrix()

    def _previous_position(self,cost_matrix):
        if len(cost_matrix) == 0:
            return

    def _create_trajectory_from_cost_matrix(self):
        particle_indices = []
        prev_lowest_cost_index = 0
        for frame_index in range(len(self._cost_matrix)):
            lowest_cost_index = np.where(
                self._cost_matrix[-frame_index - 1].T[prev_lowest_cost_index] == np.amin(self._cost_matrix[-frame_index - 1].T[prev_lowest_cost_index])
            )[0][0]
            particle_indices.append(
                lowest_cost_index
            )
            prev_lowest_cost_index = lowest_cost_index

        p = np.empty((1,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32)])
        self._trajectory = Trajectory()
        for frame_index, particle_index in enumerate(particle_indices[::-1]):
            p['frame_index'] = frame_index + self.start_point[0]
            p['time'] = self._time[frame_index + self.start_point[0]]
            p['position'] = self._particle_positions[frame_index][particle_index]
            self._trajectory._append_position(p)

    def _update_averaged_intensity(self):
        if self.boxcar_width == 0:
            self._averaged_intensity = self._frames
        else:
            self._averaged_intensity = np.empty(self._frames.shape)
            kernel = Box1DKernel(self.boxcar_width)
            for row_index, row_intensity in enumerate(self._frames):
                self._averaged_intensity[row_index] = convolve(row_intensity, kernel)

    def _calculate_second_order_intensity_moment(self, position, frame_index):
        position = int(round(position))
        if self._integration_radius_of_intensity_peaks == 0:
            return 0
        elif position == 0:
            if self._integration_radius_of_intensity_peaks == 1:
                return 2 * self._averaged_intensity[frame_index, 1] / self._calculate_first_order_intensity_moment(position, frame_index)
            else:
                second_order_index_array = np.arange(0, self._integration_radius_of_intensity_peaks + 1) ** 2
                return (
                               2 * np.dot(self._averaged_intensity[frame_index, :self.integration_radius_of_intensity_peaks + 1], second_order_index_array)
                       ) / self._calculate_first_order_intensity_moment(position, frame_index)
        elif position == self._frames.shape[1] - 1:
            if self._integration_radius_of_intensity_peaks == 1:
                return 2 * self._averaged_intensity[frame_index, -2] / self._calculate_first_order_intensity_moment(position, frame_index)
            else:
                second_order_index_array = np.arange(-self._integration_radius_of_intensity_peaks, 0) ** 2
                return 2 * np.dot(self._averaged_intensity[frame_index, -self._integration_radius_of_intensity_peaks - 1:-1],
                                  second_order_index_array) / self._calculate_first_order_intensity_moment(position, frame_index)
        elif position < self._integration_radius_of_intensity_peaks:
            w = self._integration_radius_of_intensity_peaks - position
            if w == 1:
                second_order_index_array = np.arange(-position, position + 1) ** 2
                return (
                        (
                                np.dot(self._averaged_intensity[frame_index, :2 * position + 1], second_order_index_array) +
                                2 * self._integration_radius_of_intensity_peaks ** 2 * self._averaged_intensity[frame_index, position + self._integration_radius_of_intensity_peaks]
                        ) / self._calculate_first_order_intensity_moment(position, frame_index)
                )
            else:
                second_order_index_array = np.arange(-position, position + 1) ** 2
                second_order_index_array_big = np.arange(position + 1, position + self._integration_radius_of_intensity_peaks) ** 2
                return (
                               np.dot(second_order_index_array, self._averaged_intensity[frame_index, :2 * position + 1]) +
                               2 * np.dot(second_order_index_array_big,
                                          self._averaged_intensity[frame_index, 2 * position + 1:2 * position + self._integration_radius_of_intensity_peaks])
                       ) / self._calculate_first_order_intensity_moment(position, frame_index)
        elif position > self._frames.shape[1] - 1 - self._integration_radius_of_intensity_peaks:
            w = self._integration_radius_of_intensity_peaks - (self._frames.shape[1] - position - 1)
            if w == 1:
                second_order_index_array = np.arange(-(self._frames.shape[1] - 1 - position), self._frames.shape[1] - position) ** 2
                return (
                               np.dot(second_order_index_array, self._averaged_intensity[frame_index, 2 * position - self._frames.shape[1] + 1:])
                               + 2 * self._integration_radius_of_intensity_peaks ** 2 * self._averaged_intensity[
                                   frame_index, position - self._integration_radius_of_intensity_peaks]
                       ) / self._calculate_first_order_intensity_moment(position, frame_index)
            else:
                second_order_index_array = np.arange(-(self._frames.shape[1] - 1 - position), self._frames.shape[1] - position) ** 2
                second_order_index_array_big = np.arange(- self._integration_radius_of_intensity_peaks, -(self._frames.shape[1] - position) + 1) ** 2
                return (
                               np.dot(second_order_index_array, self._averaged_intensity[frame_index, -2 * (self._frames.shape[1] - position) + 1:]) +
                               2 * np.dot(second_order_index_array_big,
                                          self._averaged_intensity[frame_index, position - self._integration_radius_of_intensity_peaks:-2 * (self._frames.shape[1] - position) + 1])
                       ) / self._calculate_first_order_intensity_moment(position, frame_index)

        else:
            w = self._integration_radius_of_intensity_peaks
            second_order_index_array = np.arange(-w, w + 1) ** 2
            return np.dot(self._averaged_intensity[frame_index, position - w:position + w + 1], second_order_index_array) / self._calculate_first_order_intensity_moment(position,
                                                                                                                                                                         frame_index)

    def _calculate_first_order_intensity_moment(self, position, frame_index):
        position = int(round(position))
        if self._integration_radius_of_intensity_peaks == 0:
            return self._averaged_intensity[frame_index, position]
        elif position == 0:
            if self._integration_radius_of_intensity_peaks == 1:
                return self._averaged_intensity[frame_index, position] + 2 * self._averaged_intensity[frame_index, position + 1]
            else:
                return (
                        self._averaged_intensity[frame_index, position] +
                        2 * np.sum(self._averaged_intensity[frame_index, position + 1:position + 1 + self._integration_radius_of_intensity_peaks])
                )
        elif position == self._frames.shape[1] - 1:
            if self._integration_radius_of_intensity_peaks == 1:
                return self._averaged_intensity[frame_index, position] + 2 * self._averaged_intensity[frame_index, position - 1]
            else:
                return (
                        self._averaged_intensity[frame_index, position] +
                        2 * np.sum(self._averaged_intensity[frame_index, position - self._integration_radius_of_intensity_peaks:position])
                )
        elif position < self._integration_radius_of_intensity_peaks:
            w = self._integration_radius_of_intensity_peaks - position
            if w == 1:
                return np.sum(self._averaged_intensity[frame_index, :2 * position + 1]) + 2 * self._averaged_intensity[frame_index, position + w + 1]
            else:
                return (
                        np.sum(self._averaged_intensity[frame_index, :2 * position + 1]) +
                        2 * np.sum(self._averaged_intensity[frame_index, 2 * position + 1:2 * position + w + 1])
                )
        elif position > self._frames.shape[1] - 1 - self._integration_radius_of_intensity_peaks:
            w = self._integration_radius_of_intensity_peaks - (self._frames.shape[1] - position - 1)
            if w == 1:
                return (
                        np.sum(self._averaged_intensity[frame_index, -2 * (self._frames.shape[1] - position) + 1:]) +
                        2 * self._averaged_intensity[frame_index, -2 * (self._frames.shape[1] - position)]
                )
            else:
                return (
                        np.sum(self._averaged_intensity[frame_index, -2 * (self._frames.shape[1] - position) + 1:]) +
                        2 * np.sum(self._averaged_intensity[frame_index, position - self._integration_radius_of_intensity_peaks:position - 1])
                )
        else:
            w = self._integration_radius_of_intensity_peaks
            return np.sum(self._averaged_intensity[frame_index, position - w:position + w + 1])

    @staticmethod
    def _find_local_maximas(y):
        local_maximas = np.where(np.r_[True, y[1:] > y[:-1]] & np.r_[y[:-1] > y[1:], True])
        return local_maximas[0].astype(np.float32)

    @staticmethod
    def _validate_class_arguments(frames, time, automatic_update):
        ParticleTracker._test_if_frames_have_correct_format(frames)
        ParticleTracker._test_if_time_has_correct_format(time)
        ParticleTracker._test_if_time_and_frames_has_same_length(time, frames)
        ParticleTracker._test_if_automatic_update_has_correct_format(automatic_update)

    @staticmethod
    def _test_if_frames_have_correct_format(frames):
        if type(frames) is not np.ndarray:
            raise TypeError('Class argument frames not of type np.ndarray')
        if not (len(frames.shape) == 2 and frames.shape[0] > 1 and frames.shape[1] > 2):
            raise ValueError('Class argument frames need to be of shape (nFrames,nPixels) with nFrames > 1 and nPixels >2')
        if not (np.max(frames.flatten()) == 1 and np.min(frames.flatten()) == 0):
            raise ValueError('Class argument frames not normalised. Max value of frames should be 1 and min value should be 0.')

        return True

    @staticmethod
    def _test_if_time_has_correct_format(time):
        if type(time) is not np.ndarray:
            raise TypeError('Class argument frames not of type np.ndarray')
        if not (len(time.shape) == 1 and time.shape[0] > 1):
            raise ValueError('Class argument time need to be of shape (nFrames,) with nFrames > 1.')
        if not all(np.diff(time) > 0):
            raise ValueError('Class argument time not increasing monotonically.')
        return True

    @staticmethod
    def _test_if_time_and_frames_has_same_length(time, frames):
        if not time.shape[0] == frames.shape[0]:
            raise ValueError('Class arguments time and frames does not of equal length.')
        return True

    @staticmethod
    def _test_if_automatic_update_has_correct_format(automatic_update):
        if not type(automatic_update) == bool:
            raise ValueError('Class argument automatic_update must be True or False.')
        return True
