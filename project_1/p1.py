# Victor Ferreira Ferrari, RA 187890
# MC920 - Introduction to Image Digital Processing
# Last modified: 08/09/2019

from cv2 import imwrite, imread, copyMakeBorder, BORDER_CONSTANT
from cv2 import imshow, waitKey, destroyAllWindows, IMREAD_COLOR, IMREAD_GRAYSCALE
from math import floor
from os.path import split, basename, join
import numpy as np
import argparse
import constants

# Self-explanatory
def show_image(img, name='Image'):
    imshow(name, img)
    waitKey(0)
    destroyAllWindows()

def save_image(img, path, folder, dist, color=True):
    color_type = 'color' if color else 'gray'
    name = color_type + '_' + dist.lower() + '_' + basename(path)
    imwrite(join(folder, name), img)

# Dithering function
# Receives the image and the error distribution (mask)
def dithering(img, error_dist, alt):
    
    # Variables for border and error diffusion
    sizex = floor(error_dist.shape[1]/2)
    sizey = error_dist.shape[0] - 1
    rev_err = np.fliplr(error_dist)
    
    # Create border on image
    img = copyMakeBorder(img, top=0, bottom=sizey, left=sizex, right=sizex, borderType = BORDER_CONSTANT, value = 0)
    img = img.astype(float, copy=False)
    
    # Half-toned image
    endX = img.shape[1] - sizex
    endY = img.shape[0] - sizey
    out = np.zeros((img.shape[0],img.shape[1]))
    
    # Apply to the entire image
    for y in range(endY):
        
        # Reverse if odd indexed line
        reverse = (y % 2 == 1) if alt else False
        curr_err = rev_err if reverse else error_dist

        for x in range(sizex, endX)[::-1 if reverse else 1]:
            img[y,x] = max(0, img[y,x])
            out[y,x] = 255 * floor(img[y,x]/128)
            
            # Error diffusion
            diff = img[y,x] - out[y,x]
            slic = img[y:y+sizey+1, x-sizex:x+sizex+1]
            slic+= (curr_err*diff)
    
    # Removing border
    out = out[:endY,sizex:endX]
    
    return out
            
# Halftones an image via dithering with error diffusion.
def halftoning(filename, error_dist, alternating=True, color=True, folder='Outputs'):
    error_dist = np.array(error_dist)
    
    # Choosing grayscale or color image
    if color:
        img = imread(filename, IMREAD_COLOR)
        for i in range(img.shape[2]):
            img[:,:,i] = dithering(img[:,:,i], error_dist, alternating)
    else:
        img = imread(filename, IMREAD_GRAYSCALE)
        img = dithering(img, error_dist, alternating)

    return img

# Parses arguments, chooses distribution and calls halftoning function
def main():
    parser = argparse.ArgumentParser(description='Halftones image according to a specific error distribution')
    parser.add_argument('file', help='Name of the file containing the image (PNG)')
    parser.add_argument('dist', help='Name of the error diffusion distribution: floyd, stevenson, burkes, sierra, stucki, jarvis.')
    parser.add_argument('--zig', metavar='True/False', type=bool, help='True if to zigzag through the image (defaults to True).', default=True)
    parser.add_argument('--color', metavar='True/False', type=bool, help='True if image is colored (defaults to True).', default=True)
    parser.add_argument('--folder', help='Folder to save halftone image (defaults to Outputs/).', default='Outputs')
    args = parser.parse_args()
    
    # Selecting error diffusion distribution
    dist = constants.select_dist(args.dist)
    
    # Halftones image
    img = halftoning(args.file, dist, args.zig, args.color, args.folder)
    
    # Shows and saves image
    show_image(img, basename(args.file))
    save_image(img, args.file, args.folder, args.dist, args.color)

if __name__ == "__main__":
    main()
