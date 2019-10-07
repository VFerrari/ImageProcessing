# Victor Ferreira Ferrari, RA 187890
# MC920 - Introduction to Image Digital Processing
# Project 3 - Properties of Objects in Images
# Last modified: 07/10/2019

from cv2 import imwrite, imread
from cv2 import imshow, waitKey, destroyAllWindows, IMREAD_COLOR
from os.path import join, basename, splitext
from sys import argv
import numpy as np
import matplotlib.pyplot as plt

def main():
    if len(argv) < 2:
        print("Please pass an image as argument!")
    
    img = imread(argv[1], IMREAD_COLOR)
    img = rgb_to_gray(img)
    show_image(img, name='Grayscale Image')
    img = gray_to_binary(img)
    show_image(img, name='Binary Image')

# Receives a RGB image and converts to grayscale.
def rgb_to_gray(img):
    return img.dot([0.0722,0.7152,0.2126]).astype(np.uint8)

# Global thresholding for binary image
def gray_to_binary(img):
    return np.where(img < 250, 0, 255).astype(np.uint8)

# Self-explanatory
def show_image(img, name='Image'):
    imshow(name, img)
    waitKey(0)
    destroyAllWindows()

def save_image(img, path, folder):
    name = 'regions' + '_' + basename(path)
    imwrite(join(folder, name), img)

if __name__ == "__main__":
    main()
