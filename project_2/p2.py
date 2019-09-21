# Victor Ferreira Ferrari, RA 187890
# MC920 - Introduction to Image Digital Processing
# Project 2 - Global and Local Thresholding on Monochromatic Images
# Last modified: 21/09/2019

from cv2 import imwrite, imread
from cv2 import imshow, waitKey, destroyAllWindows, IMREAD_GRAYSCALE
from os.path import join, basename
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
def local_thresh(img, method_key, size):
    methods = {'bernsen':bernsen}
    
    # Gets method function.
    method = methods.get(method_key, None)
    
    return img

def bernsen(img, size):
    return

# Self-explanatory
def show_image(img, name='Image'):
    imshow(name, img)
    waitKey(0)
    destroyAllWindows()

def save_image(img, path, method, folder):
    name = method.lower() + '_' + basename(path)
    imwrite(join(folder, name), img)
