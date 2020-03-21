import numpy as np
import matplotlib.pyplot as plt


class Trajectory:
    """
    Object that describes a trajectory. With functions for checking if the trajectory describes real diffusion,
    convenient plotting and calculations of diffusion coefficients.

    Parameters
    ----------
    pixel_width: float
        Defines the length one pixel corresponds to. This value will be used when calculating diffusion
        coefficients. Default is 1.

    Attributes
    ----------
    pixel_width
    particle_positions
    """

    def __init__(self, pixel_width=1):
        self._particle_positions = np.empty((0,), dtype=[('frame_index', np.int16), ('time', np.float32), ('position', np.float32)])
        self._velocities = np.empty((0, 0), dtype=np.float32)
        self._time_steps = np.empty((0, 0), dtype=np.int16)
        self._position_steps = np.empty((0, 0), dtype=np.int16)
        self._pixel_width = pixel_width

    def __add__(self, other):
        new_trajectory = Trajectory(pixel_width=self.pixel_width)
        if self.pixel_width != other.pixel_width:
            raise ValueError('Pixel width must be equal when adding trajectories together.')
        elif self._particle_positions.shape[0] == 0 and other._particle_positions.shape[0] == 0:
            return new_trajectory
        elif self._particle_positions.shape == (0,):
            new_trajectory._particle_positions = other._particle_positions
        elif other._particle_positions.shape == (0,):
            new_trajectory._particle_positions = self._particle_positions
        elif other._particle_positions[0]['frame_index'] == self._particle_positions[0]['frame_index']:
            raise ValueError('Both trajectories cant start at same frame index.')
        elif other._particle_positions.shape[0] == 1 and self._particle_positions.shape[0] == 1:
            if other._particle_positions['frame_index'][0] < self._particle_positions['frame_index'][0]:
                new_trajectory._particle_positions = np.append(other._particle_positions, self._particle_positions)
            else:
                new_trajectory._particle_positions = np.append(self._particle_positions, other._particle_positions)
        elif other._particle_positions['frame_index'][0] < self._particle_positions['frame_index'][0]:
            index = np.where(
                other._particle_positions['frame_index'] < self._particle_positions[0]['frame_index']
            )
            new_trajectory._particle_positions = np.append(other._particle_positions[index], self._particle_positions)
        elif self._particle_positions['frame_index'][0] < other._particle_positions['frame_index'][0]:
            index = np.where(
                self._particle_positions['frame_index'] < other._particle_positions[0]['frame_index']
            )
            new_trajectory._particle_positions = np.append(self._particle_positions[index], other._particle_positions)

        return new_trajectory

    @property
    def pixel_width(self):
        """
        float:
            Defines the length one pixel corresponds to. This value will be used when calculating diffusion
            coefficients. Default is 1.
        """
        return self._pixel_width

    @pixel_width.setter
    def pixel_width(self, width):
        self._pixel_width = width

    @property
    def particle_positions(self):
        """
        np.array:
            Numpy array with all particle positions in the trajectory on the form `np.array((nParticles,), dtype=[('frame_index', np.int16),
            ('time', np.float32),('position', np.int16)])`
        """
        return self._particle_positions

    def plot_trajectory(self, ax=None, **kwargs):
        """
        Plots the trajectory using the frame index and the particle position in pixels.

        ax: matplotlib axes instance
            The axes which you want the frames to plotted on. If none is provided a new instance will be created.
        **kwargs:
            Plot settings, any settings which can be used in matplotlib.pyplot.plot method.

        Returns
        -------
            matplotlib axes instance
                Returns the axes input argument or creates and returns a new instance of a matplotlib axes object.
        """
        if ax is None:
            ax = plt.axes()
        ax.plot(self._particle_positions['position'], self._particle_positions['frame_index'], np.ones((1,)), **kwargs)
        return ax

    def plot_velocity_auto_correlation(self, ax=None, **kwargs):
        """
        Plots the particle velocity auto correlation function which can be used for examining if the trajectory
        describes free diffusion.

        ax: matplotlib axes instance
            The axes which you want the frames to plotted on. If none is provided a new instance will be created.
        **kwargs:
            Plot settings, any settings which can be used in matplotlib.pyplot.plot method.

        Returns
        -------
            matplotlib axes instance
                Returns the axes input argument or creates and returns a new instance of a matplotlib axes object.
        """
        if ax is None:
            ax = plt.axes()
        ax.acorr(self._velocities, **kwargs)
        return ax

    def calculate_mean_square_displacement_function(self):
        """
        Returns
        -------
            time: np.array
                The time corresponding to the mean squared displacements.

            msd: np.array
                The mean squared displacements of the trajectory.
        """

        mean_square_displacements = np.zeros((self.particle_positions['frame_index'][-1] - self.particle_positions['frame_index'][0] + 1,),
                                             dtype=[('msd', np.float32), ('nr_of_values', np.int16)])
        times = np.arange(0, self.particle_positions['frame_index'][-1] - self.particle_positions['frame_index'][0] + 1, dtype=np.float32) * self._calculate_time_step()

        for first_index, first_position in enumerate(self.particle_positions[:-1]):
            for second_index, second_position in enumerate(self.particle_positions[first_index + 1:]):
                index_difference = second_position['frame_index'] - first_position['frame_index']
                mean_square_displacements['msd'][index_difference] += ((second_position['position'] - first_position['position']) * self.pixel_width) ** 2
                mean_square_displacements['nr_of_values'][index_difference] += 1

        for index, msd in enumerate(mean_square_displacements):
            if mean_square_displacements['nr_of_values'][index] != 0:
                mean_square_displacements['msd'][index] = msd['msd'] / mean_square_displacements['nr_of_values'][index].astype(np.float32)

        non_zeros_indices = np.nonzero(mean_square_displacements['nr_of_values'])
        return times[non_zeros_indices], mean_square_displacements['msd'][non_zeros_indices]

    def _append_position(self, particle_position):
        self._particle_positions = np.append(self._particle_positions, particle_position, axis=0)

    def _position_exists_in_trajectory(self, particle_position):
        for p in self._particle_positions:
            if np.array_equal(p, particle_position):
                return True

    def _calculate_particle_velocities(self):
        self._time_steps = np.diff(self._particle_positions['time'])
        self._position_steps = np.diff(self._particle_positions['position'] * self.pixel_width)
        self._velocities = self._position_steps / self._time_steps

    @staticmethod
    def _remove_non_unique_values(array):
        return np.unique(array)

    @staticmethod
    def _sort_values_low_to_high(array):
        return np.sort(array)

    def calculate_diffusion_coefficient_from_mean_square_displacement_function(self, fit_range=None):
        """
        Fits a straight line to the mean square displacement function and calculates the diffusion coefficient from the
        gradient of the line. The mean squared displacement of the particle position is proportional to :math:`2Dt`
        where :math:`D` is the diffusion coefficient and :math:`t` is the time.

        fit_range: list, None (default)
            Define the range of the fit, the data for the fit will be `time[fit_range[0]:fit_range[1]`` and `mean_squared_displacement[fit_range[0]:fit_range[1]]`.

        Returns
        -------
            diffusion_coefficient: float
                todo
            error: float
                todo
        """
        time, mean_square_displacement = self.calculate_mean_square_displacement_function()
        if fit_range is None:
            polynomial_coefficients, error_estimate = self._fit_straight_line_to_data(time, mean_square_displacement)
        else:
            polynomial_coefficients, error_estimate = self._fit_straight_line_to_data(time[fit_range[0]:fit_range[1]], mean_square_displacement[fit_range[0]:fit_range[1]])
        return polynomial_coefficients[0] / 2, error_estimate[0] / 2

    @staticmethod
    def _fit_straight_line_to_data(x, y):
        polynomial_coefficients, covariance_matrix = np.polyfit(x, y, 1, cov=True)
        error_estimate = [np.sqrt(covariance_matrix[0, 0]), np.sqrt(covariance_matrix[1, 1])]
        return polynomial_coefficients, error_estimate

    def plot_parts_of_trajectory_used_in_the_covariance_based_estimator_for_diffusion_coefficient(self, ax=None, **kwargs):
        """
        Plots the part of the trajectory that is used when calculating the diffusion coefficient with the covariance based estimator.

        ax: matplotlib axes instance
            The axes which you want the frames to plotted on. If none is provided a new instance will be created.
        **kwargs:
            Plot settings, any settings which can be used in matplotlib.pyplot.plot method.

        Returns
        -------
            matplotlib axes instance
                Returns the axes input argument or creates and returns a new instance of a matplotlib axes object.
        """
        if ax is None:
            ax = plt.axes()
        for index, first_position in enumerate(self.particle_positions[:-2]):
            second_position = self.particle_positions[index + 1]
            third_position = self.particle_positions[index + 2]
            if first_position['frame_index'] - second_position['frame_index'] == -1 and first_position['frame_index'] - third_position['frame_index'] == -2:
                x = [first_position['position'], second_position['position']]
                y = [first_position['frame_index'], second_position['frame_index']]
                ax.plot(x, y, np.ones((1,)), **kwargs)
        return ax

    def calculate_diffusion_coefficient_using_covariance_based_estimator(self, R=None):
        """
        Unbiased estimator of the diffusion coefficient. More info at `https://www.nature.com/articles/nmeth.2904`

        Returns
        -------
            diffusion_coefficient: float
        """
        squared_displacements = []
        covariance_term = []

        for index, first_position in enumerate(self.particle_positions[:-2]):
            second_position = self.particle_positions[index + 1]
            third_position = self.particle_positions[index + 2]
            if first_position['frame_index'] - second_position['frame_index'] == -1 and first_position['frame_index'] - third_position['frame_index'] == -2:
                squared_displacements.append((self.pixel_width * (second_position['position'] - first_position['position'])) ** 2)
                covariance_term.append((second_position['position'] - first_position['position']) * (
                        third_position['position'] - second_position['position']) * self.pixel_width ** 2)

        time_step = self._calculate_time_step()
        number_of_points_used = len(squared_displacements)
        diffusion_coefficient = np.mean(squared_displacements) / (2 * time_step) + np.mean(covariance_term) / time_step

        if R is not None:
            localisation_error = R * np.mean(squared_displacements) + (2 * R - 1) * np.mean(covariance_term)
            epsilon = localisation_error ** 2 / (diffusion_coefficient * time_step) - 2 * R
            variance_estimate = diffusion_coefficient ** 2 * ((6 + 4 * epsilon + 2 * epsilon ** 2) / number_of_points_used + 4 * (1 + epsilon) ** 2 / (number_of_points_used ** 2))
            return diffusion_coefficient, variance_estimate

        return diffusion_coefficient

    def _calculate_time_step(self):
        return (self.particle_positions['time'][1] - self.particle_positions['time'][0]) / (self.particle_positions['frame_index'][1] - self.particle_positions['frame_index'][0])

    def calculate_number_of_missing_data_points(self):
        """
        Calculates the number of frames which the particle is not found in.

        Returns
        -------
            number: int
        """
        return int((self._particle_positions['frame_index'][-1] - self._particle_positions['frame_index'][0]) - self._particle_positions['frame_index'].shape[0] + 1)

    def calculate_number_of_particle_positions_with_single_time_step_between(self):
        """
        Calculates how many times in the trajectory the particle position in found in consecutive frames..

        Returns
        -------
            number: int
        """
        return int(np.sum([diff == 1 for diff in np.diff(self._particle_positions['frame_index'])]))
