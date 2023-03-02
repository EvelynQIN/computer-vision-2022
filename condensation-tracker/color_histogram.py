import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    """
    :params:
        xmin: x-coor of the top-left corner of the bounding box
        ymin: y-coor of the top-left corner of the bounding box
        xmax: x-coor of the bottom-right corner of the bounding box
        ymax: y-coor of the bottom-right corner of the bounding box
        frame: the current frame
        hist_bin: int, number of hist bins
    :return:
        histogram: 1d array of counts for all 3 channels, (3 * hist_bin,)
    """
    # clip px & py to ensure the whole bounding box is within the frame
    h, w, _ = frame.shape

    xmin = int(np.clip(xmin, 0, w-1))
    ymin = int(np.clip(ymin, 0, h-1))
    xmax = int(np.clip(xmax, 0, w-1))
    ymax = int(np.clip(ymax, 0, h-1))

    bouding_box = frame[ymin : ymax, xmin : xmax, :] # b_width x b_hight x 3
    bins = np.linspace(start = 0, stop = 255, num = hist_bin + 1) # fix all the bin edges regardless of the frame

    histogram = np.zeros((3, hist_bin))
    histogram[0], _ = np.histogram(bouding_box[:, :, 0], bins = bins) # binning the R channel
    histogram[1], _ = np.histogram(bouding_box[:, :, 1], bins = bins) # binning the G channel
    histogram[2], _ = np.histogram(bouding_box[:, :, 2], bins = bins) # binning the B channel

    # histogram, _ = np.histogramdd(bouding_box.reshape(-1, 3), bins = (bins, bins, bins), density = True) # (hist_bin, hist_bin, hist_bin)

    return histogram.reshape(-1) / np.sum(histogram) # normalization



