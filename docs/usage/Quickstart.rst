Quick start
===========

Particle tracking
-----------------
.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from particle_tracker_one_d import ParticleTracker

    # Import the frames and the time data
    frames = np.load('examples/frames.npy')
    time = np.load('examples/time.npy')

    # Normalise the intensity
    frames_normalised = ParticleTracker.normalise_intensity(frames)

    # Create a particle tracker instance
    pt = ParticleTracker(frames=frames_normalised, time=time)

    # Set the properies of the particle tracker
    pt.boxcar_width = 10
    pt.change_cost_coefficients(1,1,0,6)
    pt.integration_radius_of_intensity_peaks = 10
    pt.particle_detection_threshold = 0.6
    pt.maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = 5
    pt.maximum_distance_a_particle_can_travel_between_frames = 40


    # Create a figure
    plt.figure(figsize=(8,20))
    ax=plt.axes()

    # Plot the kymograph
    pt.plot_all_frames(ax=ax, aspect='auto')

    # Plot all the trajectories
    for t in pt.trajectories:
        t.plot_trajectory(x='position',y='frame_index',ax=ax, marker='o')


Trajectory analysis
-------------------
.. code-block:: python

    # Select one of the trajectories
    trajectory = pt.trajectories[0]

    # Set the pixel width
    trajectory.pixel_width = 5e-4

    # Create a figure
    plt.figure(figsize=(8,8))
    ax=plt.axes()

    # Plot the velocity auto correlation function to make sure particle steps are uncorrelated
    trajectory.plot_velocity_auto_correlation(ax=ax)

    # Calculate the diffusion coefficient using a covariance based estimator
    print(trajectory.calculate_diffusion_coefficient_using_covariance_based_estimator())

Shortest path finder
--------------------
.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from particle_tracker_one_d import ShortestPathFinder

    # Import the frames and the time data
    frames = np.load('examples/frames.npy')
    time = np.load('examples/time.npy')

    # Normalise the intensity
    frames_normalised = ParticleTracker.normalise_intensity(frames)

    # Create a shortest path finder instance
    spf = ShortestPathFinder(frames=frames_normalised, time=time)

    # Set the properies of the path finder
    spf.boxcar_width = 5
    spf.integration_radius_of_intensity_peaks = 20

    # Set start point
    spf.start_point = (0,90)

    # Set end point
    spf.end_point = (232,2)

    fig = plt.figure(figsize=(5,15))
    ax = plt.axes()

    # Plot the frames and the trajectory
    spf.plot_all_frames(ax=ax, aspect='auto')
    spf.trajectory.plot_trajectory(x='position', y='frame_index',ax=ax, marker='o')

    ax.set_ylim([232, 90])
    fig.tight_layout()