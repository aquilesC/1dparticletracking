import numpy as np
import matplotlib.pyplot as plt


class Trajectory:

    def __init__(self, time_step=1, position_step=1):
        self._particle_positions = np.empty((0, 2), dtype=np.int16)
        self._velocities = np.empty((0, 0), dtype=np.float32)
        self._diffusion_coefficient = 0
        self._time_steps = np.empty((0, 0), dtype=np.int16)
        self._position_steps = np.empty((0, 0), dtype=np.int16)
        self._time_step = time_step
        self._position_step = position_step
        self._hindrance_factor = 1
        self._channel_x_dimension = None
        self._channel_y_dimension = None
        self._molecule_radius = None

    @property
    def hindrance_factor(self):
        if self._channel_x_dimension and self._channel_y_dimension and self._molecule_radius:
            return self._calculate_hindrance_factor()
        else:
            return 1

    @property
    def time_step(self):
        return self._time_step

    @time_step.setter
    def time_step(self, time):
        self._time_step = time

    @property
    def position_step(self):
        return self._position_step

    @position_step.setter
    def position_step(self, step):
        self._position_step = step

    @property
    def particle_positions(self):
        return self._particle_positions

    @property
    def channel_x_dimension(self):
        return self._channel_x_dimension

    @channel_x_dimension.setter
    def channel_x_dimension(self, dimension):
        self._channel_x_dimension = dimension

    @property
    def channel_y_dimension(self):
        return self._channel_y_dimension

    @channel_y_dimension.setter
    def channel_y_dimension(self, dimension):
        self._channel_y_dimension = dimension

    @property
    def molecule_radius(self):
        return self._molecule_radius

    @molecule_radius.setter
    def molecule_radius(self, radius):
        self._molecule_radius = radius

    def _calculate_equilibrium_partition_coefficient(self):
        if self._channel_x_dimension and self._channel_y_dimension and self._molecule_radius:
            return (1 - self._molecule_radius / self._channel_x_dimension) * (1 - self._molecule_radius / self._channel_x_dimension)
        else:
            raise ValueError('Channel dimensions or molecule radius has not been set.')

    def append_position(self, particle_position):
        self._particle_positions = np.append(self._particle_positions, particle_position.reshape((1, 2)), axis=0)

    def position_exists_in_trajectory(self, particle_position):
        for p in self._particle_positions:
            if p[0] == particle_position[0] and p[1] == particle_position[1]:
                return True

    def _calculate_particle_velocities(self):
        for index in range(0, self._particle_positions.shape[0] - 1):
            if self._particle_positions[index + 1][0] - self._particle_positions[index][0] < 5:
                time_difference = (self._particle_positions[index + 1][0] - self._particle_positions[index][0]) * self._time_step
                position_difference = (self._particle_positions[index + 1][1] - self._particle_positions[index][1]) * self._position_step
                velocity = position_difference / time_difference
                self._velocities = np.append(self._velocities, velocity)
                self._time_steps = np.append(self._time_steps, time_difference)
                self._position_steps = np.append(self._position_steps, position_difference)

    def plot_trajectory(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.axes()
        ax.plot(self._particle_positions[:, 1], self._particle_positions[:, 0], np.ones((1,)), **kwargs)
        return ax

    def plot_velocity_histogram(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.axes()
        ax.hist(self._velocities, **kwargs)
        return ax

    def plot_velocity_auto_correlation(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.axes()
        ax.acorr(self._velocities, **kwargs)
        return ax

    @staticmethod
    def _remove_non_unique_values(array):
        return np.unique(array)

    @staticmethod
    def _sort_values_low_to_high(array):
        return np.sort(array)

    def _find_time_values_for_mean_square_displacement_function(self):
        times = []
        for index, first_position in enumerate(self._particle_positions[:-1]):
            for second_position in self._particle_positions[index + 1:]:
                if first_position[0] != second_position[0]:
                    times.append(second_position[0] - first_position[0])
        times = np.array(times, dtype=np.int16)
        times = self._remove_non_unique_values(times)
        return self._sort_values_low_to_high(times)

    def _initialise_dictionary_for_mean_square_displacement_function(self):
        initial_dictionary = {}
        times = self._find_time_values_for_mean_square_displacement_function()
        for t in times:
            initial_dictionary[str(t)] = None
        return initial_dictionary

    def calculate_mean_square_displacement_function(self):
        mean_square_displacement = self._initialise_dictionary_for_mean_square_displacement_function()
        for key in mean_square_displacement.keys():
            time = int(key)
            mean_square_displacement[key] = self._calculate_mean_square_displacement_at_time(time)

        times = np.array([int(key) * self.time_step for key in mean_square_displacement.keys()])
        mean_square_displacements = np.array([mean_square_displacement[key] for key in mean_square_displacement.keys()])
        return times, mean_square_displacements

    def _calculate_mean_square_displacement_at_time(self, time):
        count = 0
        mean_square_displacement = 0
        for index, first_position in enumerate(self._particle_positions[:-1]):
            for second_position in self._particle_positions[index + 1:]:
                if second_position[0] - first_position[0] == time:
                    count += 1
                    mean_square_displacement += (self.position_step * (second_position[1] - first_position[1])) ** 2
        return mean_square_displacement / count

    def _fit_straight_line_to_mean_square_displacement_function(self):
        times, mean_square_displacements = self.calculate_mean_square_displacement_function()
        polynomial_degree = 1
        polynomial_coefficients, covariance_matrix = np.polyfit(times, mean_square_displacements, polynomial_degree, cov=True)
        error_estimate = [np.sqrt(covariance_matrix[0, 0]), np.sqrt(covariance_matrix[1, 1])]
        return polynomial_coefficients, error_estimate

    def calculate_diffusion_coefficient_from_velocity(self):
        diffusion_coefficient_velocity = 0
        for index, velocity in enumerate(self._velocities):
            diffusion_coefficient_velocity += velocity ** 2 * self._time_steps[index] / 4
        return self.hindrance_factor * diffusion_coefficient_velocity / len(self._velocities)

    def calculate_diffusion_coefficient_from_mean_square_displacement_function(self):
        polynomial_coefficients, error_estimate = self._fit_straight_line_to_mean_square_displacement_function()
        return self.hindrance_factor * polynomial_coefficients[0] / 2, error_estimate[0] / 2

    def calculate_diffusion_coefficient_using_covariance_based_estimator(self):
        return 1

    def _calculate_hindrance_factor(self):
        equilibrium_partition_coefficient = self._calculate_equilibrium_partition_coefficient()
        ratio_molecule_size_and_dimension = self._molecule_radius / np.sqrt(self._channel_x_dimension * self._channel_y_dimension)
        H = 1 + 9 / 8 * ratio_molecule_size_and_dimension * np.log(ratio_molecule_size_and_dimension) - 1.56034 * ratio_molecule_size_and_dimension + \
            0.528155 * ratio_molecule_size_and_dimension ** 2 + 1.91521 * ratio_molecule_size_and_dimension ** 3 - \
            2.81903 * ratio_molecule_size_and_dimension ** 4 + 0.270788 * ratio_molecule_size_and_dimension ** 5 + \
            1.10115 * ratio_molecule_size_and_dimension ** 6 - 0.435933 * ratio_molecule_size_and_dimension ** 7
        return H / equilibrium_partition_coefficient
