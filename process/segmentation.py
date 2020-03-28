from skimage.segmentation import felzenszwalb, slic


def do_superpixel_felzenszwalb(image, scale=100):
    segments = felzenszwalb(image, scale=scale, sigma=2, multichannel=False)
    return segments


def do_superpixel_slic(image, segments=100, compactness=1):
    segments = slic(image, multichannel=False, n_segments=segments, sigma=2, compactness=compactness)
    return segments
