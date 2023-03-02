import numpy as np

def resample(particles, particles_w):
    """Sample particles of the same size according to particle_w with replacement
    """
    n_particles = particles.shape[0]
    sample_indexes = np.random.choice(n_particles, n_particles, p = particles_w, replace=True) # sample with replacement acc. to the weights
    return particles[sample_indexes], particles_w[sample_indexes] / sum(particles_w[sample_indexes]) # return the normalised weights