# Victor Ferreira Ferrari, RA 187890
# MC920 - Introduction to Image Digital Processing
# Project 2 - Global and Local Thresholding on Monochromatic Images
# Last modified: 21/09/2019

from cv2 import imwrite, imread
from cv2 import imshow, waitKey, destroyAllWindows, IMREAD_GRAYSCALE
from os.path import join, basename
from math import exp
import numpy as np

def thresholding(filename, method='global', thresh=127, neigh_size=3, folder='Outputs'):
    
    # Reads image and applies thresholding
    img = imread(filename, IMREAD_GRAYSCALE)
    if method == 'global':
        img = global_thresh(img, thresh) 
    else:
        img = local_thresh(img, method, neigh_size)
    
    # Shows and saves final image
    show_image(img, basename(filename))
    save_image(img, filename, method_key, folder)

# Global threshold: Fixed threshold value.
def global_thresh(img, thresh):
    return np.uint8(np.where(img>thresh, 255, 0))

# Local threshold: thresholds depending on neighborhood of arbitrary size.
# Size should be small enough to preserve detail, but large enough to supress noise.
def local_thresh(img, method_key, size):
    methods = {'bernsen':bernsen, 'niblack':niblack, 'sauvola':sauvola_pietaksinen,
               'more':phansalskar_more_sabale, 'contrast': contrast_method,
               'mean':mean_method, 'median':median_method}
    
    # Gets method function.
    method = methods.get(method_key, None)
    
    return img

# Local thresholding method: (zmax + zmin)/2 is the threshold
def bernsen(window, *_):
    return (np.amax(window) - np.amin(window))/2

# Local thresholding method: mean + k*stdev is the threshold. 
# Statistical method, k is an adjustment parameter.
def niblack(window, k, *_):
    return np.mean(window) + k*np.std(window)

# Local thresholding method building on Niblack.
# Particularly for images with bad lighting.
# k and R are adjustment parameters. Suggested k=0,5 and R=128
def sauvola_pietaksinen(window, k, r, *_):
    return np.mean(window) * (1 + k*(np.std(window)/r - 1))

# Local thresholding method building on Sauvola/Pietaksinen.
# Deals with low contrast images.
# k, R, p, q are adjustment parameters. Suggested k=0,25, R=0,5, p=2 and q=10.
# R is different from Sauvola because it uses normalized intensity.
def phansalskar_more_sabale(window, k, r, p, q):
    mean = np.mean(window)
    return mean * (1 + p * exp(-1*q*mean) + k * (np.std(window)/r - 1))

# Local thresholding method that checks contrast.
# Pixel is background or object if value is closest to local max or min, respectively.
def contrast_method(window, *_):
    shape = window.shape
    pixel = window[shape[0]//2, shape[1]//2]
    return 0 if abs(pixel-np.amax(window)) < abs(pixel-np.amin(window)) else 255

# Local thresholding method: mean is the threshold
def mean_method(window, *_):
    return np.mean(window)

# Local thresholding method: median is the threshold
def median_method(window, *_):
    return np.median(window)

# Self-explanatory
def show_image(img, name='Image'):
    imshow(name, img)
    waitKey(0)
    destroyAllWindows()

def save_image(img, path, method, folder):
    name = method.lower() + '_' + basename(path)
    imwrite(join(folder, name), img)
