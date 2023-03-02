import numpy as np

def propagate(particles, frame_height, frame_width, params):
    """
    :params:
        particles: ndarray in dim (n_particles, 2) if no motion,  (n_particles, 4) if velocity constant
        frame_height:
        frame_width:
        params: model parameters dict
    :returns:
        new_particles: the predicted states of particles
    """
    noise = np.random.standard_normal(particles.shape) # get random noise from a standard normal deistribution

    if params['model'] == 0: # if assume no motion
        A_T = np.array([[1, 0],
                        [0, 1]])
        
        sd = params['sigma_position']
        new_particles = particles @ A_T + noise * sd
    else: # if assume constant velocity
        A_T = np.array([[1, 0, 0, 0],  # directly compute the transposed A for matrix multiplication
                        [0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1]])
        sd_pos = params['sigma_position']
        sd_v = params['sigma_velocity']
        new_particles = particles @ A_T + noise * [sd_pos, sd_pos, sd_v, sd_v]
    
    # clip the particle centers within the valid range over the current frame
    new_particles[:, 0] = np.clip(new_particles[:, 0], 0, frame_width - 1)
    new_particles[:, 1] = np.clip(new_particles[:, 1], 0, frame_height - 1)

    return new_particles
        
        
