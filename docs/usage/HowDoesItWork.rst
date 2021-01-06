How does it work?
=================
I will here give a basic explanation of how the algorithms work. For deeper understanding of the particle tracking
algorithm, I refer you to

`1. Sbalzarini, Ivo F., and Petros Koumoutsakos., Journal of structural biology 151.2 (2005): 182-195.`__

__ https://www.sciencedirect.com/science/article/pii/S1047847705001267

and for the shortest path finder I refer you to
`2. Dijkstra's algorithm`__

__ https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm

Definitions
-----------
A particle observation or feature point :math:`p = [t,\hat{x_p}]` where :math:`t` is the frame index and :math:`\hat{x_p}` is the estimated position comes with associated
intensity moments defined as the 0th order moment

.. math::

    m_0(p) = \sum_{i^2 \leq w^2} I^t (\hat{x_p} + i)

and the 2nd order moment

.. math::

    m_2(p) = \sum_{i^2 \leq w^2} i^2 I^t (\hat{x_p} + i)

where :math:`I^t` is the intensity of frame :math:`t` and :math:`w` is the integration radius.

Particle Tracker
----------------
The particle tracker finds arbitrary number of trajectories in a kymograph. This is done by linking particle detections
together by minimising a cost describing the chance of detections being the same particle in two different frames.

Initialisation
______________
To create an instance of the particle tracker one has to provide the frames and the corresponding times.
.. code-block:: python

    from particle_tracker_one_d import ParticleTracker

    # Import frames and time data
    frames = np.load('examples/frames.npy')
    time = np.load('examples/time.npy')

    # Create a particle tracker instance
    pt = ParticleTracker(frames=frames, time=times)

Preferably the frames should be normalised according to :math:`I_{normalised} = (I-I_{min})/(I_{max}-I_{min})` but is
not a requirement.



Finding particle positions
__________________________
To detect possible particles an intensity detection threshold is set by the attribute
:code:`pt.particle_detection_threshold`. Local maximas over this threshold are considered as initial possible particle
detections. If however, two maximas are found within a distance of :code:`2 * integration_radius_of_intensity_peaks`
pixels, the lowest maxima of the two points is discarded. Each position is then refined using a centroid estimation
around the local maxima, using the same integration radius.

Linking procedure
_________________
The linking procedure is then performed as described in[1] where two frames are analysed at a time and a cost for
linking a particle in frame :math:`t` with either a particle or a dummy particle in frame :math:`t+r`, is calculated for
all combination of particles in both frames. A link matrix is then optimized to find the set of links between particles
yielding the lowest total cost, still respecting the topography requirement, that each particle in frame :math:`t` can
only be linked to one particle in frame :math:`t+r`. The maximum :math:`r` allowed is set by the attribute
:code:`pt.maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles`. The cost function
used for the associations is

.. math::
    \phi(p_1,p_2) = a \cdot (x_{p_1}-x_{p_2})^2 + b \cdot (m_0(p_1) - m_0(p_2))^2 + c \cdot (m_2(p_1) - m_2(p_2))^2 + d \cdot (r-1)^2

where the :math:`d \cdot (r-1)^2` term is added to promote the linking of particle detections closer in time. To change
the coefficients :math:`a,b,c,d` one can call the function :code:`pt.change_cost_coefficients(a=1,b=1,c=1,d=1)`. In
case of many particles there is a limit to the maximum number of pixels a particle can move between two consecutive
frames set by the instance attribute :code:`pt.maximum_distance_a_particle_can_travel_between_frames`. If this limit is
exceeded the cost is set to infinity and any linking is effectively blocked. When all link matrices are optimised the
trajectories are built. If a particle is linked to the dummy particle, the lowest cost association to a non-dummy
particle in the following frames is linked. If no such association does exist, the trajectory is ended.

Shortest Path Finder
--------------------
The intention of the Shortest Path Finder is to refine trajectories that are missing positions in several frames. This
is done by finding a path connecting a start, an end and intermediate static positions. It is based on the well known
Dijkstra's algorithm.

Initialisation
______________
To create an instance of the shortest path finder one has to provide the frames and the corresponding times.

.. code-block:: python

    from particle_tracker_one_d import ShortestPathFinder

    # Import frames and time data
    frames = np.load('examples/frames.npy')
    time = np.load('examples/time.npy')

    # Create a particle tracker instance
    spf = ShortestPathFinder(frames=frames, time=times)

Preferably the frames should be normalised according to :math:`I_{normalised} = (I-I_{min})/(I_{max}-I_{min})` but is
not a requirement.

Set the static points
_____________________
The shortest path finder needs a start and an en point. These are set by

.. code-block:: python

    spf.start_point = (start_frame, start_position)
    spf.start_point = (end_frame, end_position)

both the values should be integers and represent the indices of the start and end point in the frames. There is then a
possibility to add more points that the trajectory is forced to go through. This is done by the attribute
:code:`static_points`

.. code-block:: python

    spf.static_points =[(frame_1, position_1), (frame_2, position_2), ..., (frame_n, position_n)]

Finding particle positions
__________________________
Possible particles is then found by looking for intensity maximas over the intensity detection threshold that is set by
the attribute :code:`spf.particle_detection_threshold` in the frames between the start and end point, skipping
the frames with the static points. If however, two maximas are found within a double distance of
the attribute :code:`spf.integration_radius_of_intensity_peaks`, the lowest maxima of the two points is discarded.
Each position is then refined using a centroid estimation around the local maxima, using the same integration radius.
This also includes the start, end and static points.

Finding the shortest path
_________________________
The algorithm then finds the shortest path defined by the cost/distance between particles

.. math::
    \phi(p_1,p_2) = a \cdot (x_{p_1}-x_{p_2})^2 + b \cdot (m_0(p_1) - m_0(p_2))^2 + c \cdot (m_2(p_1) - m_2(p_2))^2

where the coefficients :math:`a,b,c` can be changed using the function :code:`spf.change_cost_coefficients(a=1,b=1,c=1)`.
The algorithm then works as follows:

1. Store all positions in arrays :math:`\{P^t\}_{t=t_0}^{t_n}` . Where :math:`t_0` and :math:`t_n` is the start and end indices of the frames.
2. Start at :math:`t=t_0` and calculate the cost between the position at :math:`t_0` and the positions at :math:`t_1`. Store these costs in a matrix :math:`C^1=c_{ij}=\phi(p_i,p_j)`. These now describe the cost to go to each position in frame :math:`t_1`.
3. Continue calculate for all :math:`n` the cost between particles in frame :math:`t_{n}` to particles in frame :math:`t_{n+1}` and add the lowest cost from the i:th column in the previous cost matrix

    .. math::
        C^n = c_{ij} = \phi(p_i,p_j) + min_{i^'}(C_{i^'i}^{n-1})

4. Find the lowest value in :math:`C^n`. This will describe the lowest possible cost from the first position to the final, passing through all the positions in the initial sparse trajectory.
5. Build the trajectory by going backwards in the cost matrices following the lowest cost path.


Trajectory
----------
The trajecory class is made for analysing the trajectories found by the particle tracker or the shoertest path finder. It
has some methods attached to it to make this easier and faster. It is possible to instanciate the trajectory class but
the intentional way is that it is delivered to the user as already instanciated objects under the attribute :code:`pt.trajectories`
and :code:`spf.trajectory`. If you want to make your own instance, it is preferably done like this

.. code-block:: python

    t = Trajectory()
    t.particle_positions = positions

positions should be a :code:`numpy` structured array with field names 'time', 'frame_index' and 'position'.

Velocity auto correlation
_________________________
A common way to check that a trajectory represents free diffusion is to plot the velocity auto correlation and check if
velocitites are correlated. This can be done with the method :code:`t.plot_velocity_auto_correlation()`.

Calculate diffusion coefficients
________________________________
The software comes with two methods of determining the diffusion coefficient, either by fitting a straight line to the
mean squared displacement function :code:`t.calculate_diffusion_coefficient_from_mean_square_displacement_function()`
or by a covariance based estimator :code:`t.calculate_diffusion_coefficient_using_covariance_based_estimator()`.
For more information about determining diffusion coefficients, I refer you to

`Optimal estimation of diffusion coefficients from single-particle trajectories.`__

__ https://journals.aps.org/pre/abstract/10.1103/PhysRevE.89.022726
