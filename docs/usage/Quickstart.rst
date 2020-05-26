Quick start
===========

1. Start by finding trajectories

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
    pt.integration_radius_of_intensity_peaks = 25
    pt.particle_detection_threshold = 0.7
    pt.maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = 5
    pt.maximum_distance_a_particle_can_travel_between_frames = 50


    # Create a figure
    plt.figure(figsize=(10,10))
    ax=plt.axes()

    # Plot the kymograph
    pt.plot_all_frames(ax=ax, aspect='auto')

    # Plot all the trajectories
    for t in pt.trajectories:
        t.plot_trajectory(ax=ax, marker='o')


2. Analyse one of the trajectories

.. code-block:: python

    # Select one of the trajectories
    trajectory = pt.trajectories[0]

    # Set the pixel width
    trajectory.pixel_width = 5e-4

    # Create a figure
    plt.figure(figsize=(10,10))
    ax=plt.axes()

    # Plot the velocity auto correlation function to make sure particle steps are uncorrelated
    trajectory.plot_velocity_auto_correlation(ax=ax, aspect='auto')

    # Calculate the diffusion coefficient using a covariance based estimator
    print(trajectory.calculate_diffusion_coefficient_using_covariance_based_estimator())
