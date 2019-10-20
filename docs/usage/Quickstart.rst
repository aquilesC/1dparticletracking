Quick start
===========

This software was developed to track particles in Kymographs, eg. intensity graphs which change in time.

code-block:: python
    import matplotlib.pyplot as plt
    import numpy as np
    from particle_tracker_one_d import ParticleTracker

    # Import the frames and the time data
    kymograph_frames = np.load('frames.npy')
    time = np.load('time.npy')

    # Normalise the intensity
    frames_normalised = PartParticleTracker.normalise_intensity(kymograph_frames)

    # Create a particle tracker instance
    pt = ParticleTracker(frames=frames_normalised, time=time)

    # Set the properies of the particle tracker
    pt.boxcar_width = 10
    pt.integration_radius_of_intensity_peaks = 25
    pt.feature_point_threshold = 0.7
    pt.maximum_number_of_frames_a_particle_can_disappear_and_still_be_linked_to_other_particles = 5
    pt.maximum_distance_a_particle_can_travel_between_frames = 20

    # Create a figure
    plt.figure(figsize=(10,10))
    ax=plt.axes()

    # Plot the kymograph
    pt.plot_all_frames(ax=ax, aspect='auto')

    # Plot all the trajectories
    for t in pt.trajectories:
        t.plot_trajectory(ax=ax, marker='o')
