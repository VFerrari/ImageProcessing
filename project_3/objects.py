# Victor Ferreira Ferrari, RA 187890
# MC920 - Introduction to Image Digital Processing
# Project 3 - Properties of Objects in Images
# Last modified: 08/10/2019

from os.path import join, basename, splitext
from sys import argv
from skimage import img_as_float, img_as_ubyte
from skimage.color import label2rgb
from skimage.measure import label, regionprops
import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    if len(argv) < 2:
        print("Please pass an image as argument!")
    
    img = cv2.imread(argv[1], cv2.IMREAD_COLOR)
    show_image(img, name='Colored Image')
    
    label = region_split(img)
    object_properties(label)
    

# Splits regions on an image.
# Image has to have gray objects on a white background.
# Return region contours.
def region_split(img):
    
    # Grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Threshold for black background and white objects.
    _, gray = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    show_image(gray, name='Binary Image')
    
    # Label image
    gray2 = img_as_float(gray)
    labeled = label(gray2)
    
    # Find object contours and show.
    cont, _= cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gray[:] = 255
    cv2.drawContours(gray, cont, -1, (0,0,255), 2)
    show_image(gray, name='Contours')
    
    return labeled

# Get object properties from labeled image.
def object_properties(label):
    regions = regionprops(label)
    
    for region in regions:
        print(region.perimeter)
        print(region.area)
    

# Self-explanatory
def show_image(img, name='Image'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(img, path, folder):
    name = 'regions' + '_' + basename(path)
    cv2.imwrite(join(folder, name), img)

if __name__ == "__main__":
    main()
