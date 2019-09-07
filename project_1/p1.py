from cv2 import imwrite, imread, copyMakeBorder, BORDER_CONSTANT
from math import floor
import os
import numpy as np

# Dithering function
# Receives the image and the error distribution (mask)
def dithering(img, error_dist, alt):
    
    # Variables for border and error diffusion
    sizex = floor(error_dist.shape[1]/2)
    sizey = error_dist.shape[0] - 1
    
    # Create border on image
    img = copyMakeBorder(img, top=0, bottom=sizey, left=sizex, right=sizex, borderType = BORDER_CONSTANT, value = 0)
    
    # Half-toned image
    shapeY,shapeX = img.shape
    endX = shapeX - sizex
    endY = shapeY - sizey
    out = np.zeros((shapeY,shapeX))
    
    # Apply to the entire image
    for y in range(endY):
        
        # Reverse if odd indexed line
        reverse = (y % 2 == 1) if alt else False
        
        for x in range(sizex, endX)[::-1 if reverse else 1]:
            out[y,x] = 255 * floor(img[y,x]/128)
            
            # Error diffusion
            diff = int(img[y,x]) - int(out[y,x])
            for i in range(sizey+1):
                for j in range(-sizex, sizex+1):
                    img[y+i,x+j]+=error_dist[i,j + sizex-1]*diff
    
    # Removing border
    out = out[:endY,sizex:endX]
    return out
            
# Halftones an image via dithering with error diffusion.
def halftoning(filename, error_dist, alternating):
    img = imread(filename, 0)
    error_dist = np.array(error_dist)
    img = dithering(img, error_dist, alternating)
    imwrite("Outputs/test.png", img)
