# Victor Ferreira Ferrari, RA 187890
# MC920 - Introduction to Image Digital Processing
# Project 2 - Global and Local Thresholding on Monochromatic Images
# Last modified: 24/09/2019

from cv2 import imwrite, imread
from cv2 import imshow, waitKey, destroyAllWindows, IMREAD_GRAYSCALE
from os.path import join, basename
from math import exp
import numpy as np
import argparse

def main():
    methods = ['global', 'bernsen', 'niblack', 'sauvola', 'more', 'contrast', 'mean', 'median']
    
    parser = argparse.ArgumentParser(description='Applies global or local thresholding to an image, according to various methods.')
    parser.add_argument('file', help='Name of the file containing the image (PGM)')
    parser.add_argument('method', help='Name of the thresholding method: global, bernsen, niblack, sauvola, more, contrast, mean, median. "all" for trying every one.')
    parser.add_argument('--size', help='Size of local thresholding window (defaults to 7, for a 7x7 window).', default=7, type=int)
    parser.add_argument('--folder', help='Folder to save binary image (defaults to Outputs/).', default='Outputs')
    parser.add_argument('--thresh', help='Value of global threshold. May not be needed, defaults to 128.', default=128, type=int)
    parser.add_argument('--k', help='Value of parameter "k". May not be needed, defaults to 0.25, for "more".', default=0.25, type=float)
    parser.add_argument('--R', help='Value of parameter "R". May not be needed, defaults to 0.5, for "more".', default=0.5, type=float)
    parser.add_argument('--p', help='Value of parameter "p". May not be needed, defaults to 2.', default=2, type=float)
    parser.add_argument('--q', help='Value of parameter "q". May not be needed, defaults to 10.', default=10, type=float)
    
    args = parser.parse_args()
    
    # Checking method
    if args.method == 'all':
        mets = methods 
    elif args.method not in methods:
        print("Not a valid method!")
        return
    else:
        mets = [args.method]
    
    # Thresholds image
    for i in range(len(mets)):
        thresholding(args.file, mets[i], args.thresh, args.size, args.folder, (args.k, args.R, args.p, args.q))

def thresholding(filename, method='global', thresh=128, neigh_size=9, folder='Outputs', params=(0.25,0.5,2,10)):
    
    # Reads image and applies thresholding
    img = imread(filename, IMREAD_GRAYSCALE)
    if method == 'global':
        img = global_thresh(img, thresh) 
    else:
        img = local_thresh(img, method, neigh_size, *params)
    
    # Shows and saves final image
    show_image(img, basename(filename))
    save_image(img, filename, method, folder)

# Global threshold: Fixed threshold value.
def global_thresh(img, thresh):
    return np.uint8(np.where(img >= thresh, 255, 0))

# Local threshold: thresholds depending on neighborhood of arbitrary size.
# Size should be small enough to preserve detail, but large enough to supress noise.
def local_thresh(img, method_key, size, k=0.25, r=0.5, p=2, q=10):
    methods = {'bernsen':bernsen, 'niblack':niblack, 'sauvola':sauvola_pietaksinen,
               'more':phansalskar_more_sabale, 'contrast': contrast_method,
               'mean':mean_method, 'median':median_method}
    
    # Gets threshold method.
    thresh = methods.get(method_key, None)
    t_mat = np.zeros(img.shape)
    deltaXY = size//2
    
    # Applies locally
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            win = img[max(0, y-deltaXY) : min(y+deltaXY+1, img.shape[0]),
                      max(0, x-deltaXY) : min(x+deltaXY+1, img.shape[1])]
            t_mat[y,x] = thresh(win, k, r, p, q)
            
    img = np.uint8(np.where(img >= t_mat, 255, 0))
    return img

# Local thresholding method: (zmax + zmin)/2 is the threshold
def bernsen(window, *_):
    return (np.amax(window) - np.amin(window))/2

# Local thresholding method: mean + k*stdev is the threshold. 
# Statistical method, k is an adjustment parameter.
# k=-0,2 is suggested, for window of size 15.
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
    return 0 if abs(int(pixel)-int(np.amax(window))) < abs(int(pixel)-int(np.amin(window))) else 255

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

if __name__ == "__main__":
    main()
