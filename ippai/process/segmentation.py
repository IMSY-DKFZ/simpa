from skimage.segmentation import slic


def do_superpixel_segmentation(image, num_segments=100):
    segments = slic(image, n_segments=num_segments, multichannel=False)
    return segments
