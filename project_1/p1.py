from cv2 import imwrite, imread
from math import floor
import os
import numpy as np

# Dithering function
# Receives the image and the error distribution (mask)
def dithering(img, error_dist, alt):
    
    # Half-toned image
    out = np.zeros((img.shape[0],img.shape[1]))
    
    # Apply to the entire image
    for y in range(img.shape[0]):
        
        # Reverse if odd indexed line
        reverse = (y % 2 == 1) if alt else False
        
        for x in range(img.shape[1])[::-1 if reverse else 1]:
            out[y,x] = 255 * floor(img[y,x]/128)
            
            # Error diffusion
            diff = int(img[y,x]) - int(out[y,x])
            sizex = floor(error_dist.shape[1]/2)
            sizey =  floor(error_dist.shape[0]/2)
            for i in range(sizey + 1):
                if y+i < img.shape[0]:
                    for j in range(-sizex, sizex+1):
                        if x+j < img.shape[1] and x+j >= 0:
                            img[y+i,x+j]+=error_dist[i,j + sizex-1]*diff
    return out
            

def halftoning(filename, error_dist, alternating):
    img = imread(filename, 0)
    img = dithering(img, error_dist, alternating)
    imwrite("Outputs/test.png", img)
