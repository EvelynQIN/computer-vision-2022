import numpy as np

def estimate(particles, particle_w):
    mean = particle_w.T @ particles # (1, 2)
    return mean