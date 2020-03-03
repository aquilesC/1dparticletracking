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
        self._minimal_shortest_path_finder = None
        self._start_point = None
        self._end_point = None
        self._shortest_path = None

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
                self._update_association_matrix()
                self._reconnect_broken_links()
                self._update_trajectories()

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
                self._update_association_matrix()
                self._reconnect_broken_links()
                self._update_trajectories()

    @property
    def start_point(self):
        """
        tuple:
            (frame_index, position_index), The start point of the path you want to find.
        """
        return self._start_point

    @start_point.setter
    def start_point(self, start_point):
        if type(start_point) not in [list, tuple, np.array]:
            raise TypeError('Start point must be list, tuple or np.array with form (frame_index, position_index)')
        else:
            for val in start_point:
                if type(val) not in [int, np.int16, np.int32]:
                    raise TypeError('Start point values must be integers')
            if not (start_point[0] == self._start_point[0] and start_point[1] == self._start_point[1]):
                self._start_point = (int(start_point[0]), int(start_point[1]))
                if self._automatic_update:
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
        if type(end_point) not in [list, tuple, np.array]:
            raise TypeError('End point must be list, tuple or np.array with form (frame_index, position_index)')
        else:
            for val in end_point:
                if type(val) not in [int, np.int16, np.int32]:
                    raise TypeError('End point values must be integers')
            if not (end_point[0] == self._end_point[0] and end_point[1] == self._end_point[1]):
                self._end_point = (int(end_point[0]), int(end_point[1]))
                if self._automatic_update:
                    self._update_shortest_path()

    @property
    def shortest_path(self):
        """
        trajectory:
            The shortest path between the start and end point, defined by the cost function.
        """
        return self._shortest_path

    def _update_shortest_path(self):
        return

    def _update_averaged_intensity(self):
        if self.boxcar_width == 0:
            self._averaged_intensity = self._frames
        else:
            self._averaged_intensity = np.empty(self._frames.shape)
            kernel = Box1DKernel(self.boxcar_width)
            for row_index, row_intensity in enumerate(self._frames):
                self._averaged_intensity[row_index] = convolve(row_intensity, kernel)

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


