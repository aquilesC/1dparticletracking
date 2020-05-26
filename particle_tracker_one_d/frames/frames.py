import numpy as np


class Frames:
    """
    Class for handling frames, with averaging and easy plot functions.

    Parameters
    ----------
    frames: np.array
        The frames in which trajectories are to be found. The shape of the np.array should be (nFrames,xPixels). The intensity of the frames should be normalised according to
        :math:`I_n = (I-I_{min})/(I_{max}-I_{min})`, where :math:`I` is the intensity of the frames, :math:`I_{min}`, :math:`I_{max}` are the global intensity minima and maxima of the
        frames.
    time: np.array
        The corresponding time of each frame.
    automatic_update: bool
        Should the class update automatically when the property boxcar_width is changed. Default=True

    Attributes
    ----------
    frames
    time
    boxcar_width
    """

    def __init__(self, frames, time, automatic_update=True):
        Frames._validate_class_arguments(frames, time, automatic_update)
        self._automatic_update = automatic_update
        self._frames = frames
        self._averaged_intensity = frames
        self.time = time
        self._boxcar_width = 0

    @property
    def frames(self):
        """
        np.array:
            The frames which have been restored by boxcar averaging.
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

    def plot_all_frames(self, ax=None, **kwargs):
        """
        ax: matplotlib axes instance
            The axes which you want the frames to plotted on. If none is provided a new instance will be created.
        **kwargs:
            Plot settings, any settings which can be used in matplotlib.pyplot.imshow method.

        Returns
        -------
            matplotlib axes instance
                Returns the axes input argument or creates and returns a new instance of an matplotlib axes object.
        """
        if ax is None:
            ax = plt.axes()
        ax.imshow(self._averaged_intensity, **kwargs)
        return ax

    def _update_averaged_intensity(self):
        if self.boxcar_width == 0 or self.boxcar_width == 1:
            self._averaged_intensity = self._frames
        else:
            self._averaged_intensity = np.empty((self._frames.shape[0],self._frames.shape[1]-self._boxcar_width+1), dtype=np.float32)
            kernel = np.ones((self.boxcar_width,)) / self.boxcar_width
            for row_index, row_intensity in enumerate(self._frames):
                self._averaged_intensity[row_index] = np.convolve(row_intensity, kernel, mode='valid')

    @staticmethod
    def _validate_class_arguments(frames, time, automatic_update):
        Frames._test_if_frames_have_correct_format(frames)
        Frames._test_if_time_has_correct_format(time)
        Frames._test_if_time_and_frames_has_same_length(time, frames)
        Frames._test_if_automatic_update_has_correct_format(automatic_update)

    @staticmethod
    def _test_if_frames_have_correct_format(frames):
        if type(frames) is not np.ndarray:
            raise TypeError('Class argument frames not of type np.ndarray')
        if not (len(frames.shape) == 2 and frames.shape[0] >= 1 and frames.shape[1] >= 1):
            raise ValueError(
                'Class argument frames need to be of shape (nFrames,nPixels) with nFrames >= 1 and nPixels >= 1')
        return True

    @staticmethod
    def _test_if_time_has_correct_format(time):
        if type(time) is not np.ndarray:
            raise TypeError('Class argument frames not of type np.ndarray')
        if not (len(time.shape) == 1 and time.shape[0] >= 1):
            raise ValueError('Class argument time need to be of shape (nFrames,) with nFrames >= 1.')
        if not all(np.diff(time) > 0) and not time.shape[0] == 1:
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
