import numpy as np
from chi2_cost import chi2_cost
from color_histogram import color_histogram

def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):
    """
    :params:
        particles: dim (n_particles, state_len)
        frame: the current frame
        bbox_height:
        bbox_width:
        hist_bin: number of bins per channel
        hist: target histogram
        sigma_observe: sigma of the Gaussian for reweighting
    :returns:
        particle_w: the weight for each particles, (n_particles, )
    """

    particle_w = np.zeros(particles.shape[0])
    scaling = 1.0 / (np.sqrt(2 * np.pi) * sigma_observe) # precompute the scaling norm of the Gaussian, since it is reused many times
    # h, w, _ = frame.shape

    # iterate over each particle
    for i in range(particles.shape[0]):
        px, py = particles[i, 0:2]

        # clip px & py to ensure the whole bounding box is within the frame
        # px = np.clip(px, 0.5 * bbox_width, w - 1 - 0.5 * bbox_width)
        # py = np.clip(px, 0.5 * bbox_height, h - 1 - 0.5 * bbox_height)
        xmin = px - 0.5 * bbox_width
        ymin = py - 0.5 * bbox_height
        xmax = px + 0.5 * bbox_width
        ymax = py + 0.5 * bbox_height
        hist_p = color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin) # compute the histogram for the sample particle
        chi_dist = chi2_cost(hist_p, hist) # compute the chi distance of the sample from the target histogram
        particle_w[i] = scaling * np.exp(- (chi_dist ** 2) / (2 * (sigma_observe ** 2))) # the weight is a Gaussian based on chi distance
    
    # normalize the weights to ensure the sum is 1
    particle_w = particle_w / np.sum(particle_w)
    return particle_w
 



    

    
