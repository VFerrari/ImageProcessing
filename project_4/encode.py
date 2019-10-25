# Victor Ferreira Ferrari, RA 187890
# MC920 - Introduction to Image Digital Processing
# Project 4 - Steganography: Encoder
# Last modified: 25/10/2019

from cv2 import imwrite, imread
from cv2 import imshow, waitKey, destroyAllWindows, IMREAD_COLOR
from os.path import join, basename
import numpy as np
import argparse

def main():
    
    # Arguments
    parser = argparse.ArgumentParser(description='Adds hidden message in an image.')
    parser.add_argument('in_img', help='Name of the file containing the image (PNG)')
    parser.add_argument('in_txt', help='Name of the file containing the text.')
    parser.add_argument('bit_plane', help='Bit plane in which to hide the message.', type=int)
    parser.add_argument('out_img', help='Name of the file containing the coded image (PNG)')
    parser.add_argument('--folder', help='Folder to save coded image (defaults to Outputs/).', default='Outputs')

    args = parser.parse_args()

    # Read color image
    img = imread(args.in_img, IMREAD_COLOR)
    
    # Read text
    with open(args.in_txt, 'r') as f:
        txt = f.read()
    
    txt = txt_to_binary(txt)
    out = steganography_encode(img, txt, args.bit_plane)    
    
    show_image(out, name='Coded')
    save_image(out, args.out_img, args.folder)

# Transforms text into ascii codes
# Transform ascii codes into binary
def txt_to_binary(txt):
    
    # Vectorize functions
    list_v = np.vectorize(list)
    ord_v = np.vectorize(ord)
    bin_v = np.vectorize(np.binary_repr)
    
    # Transform into ascii codes.
    arr = list_v(txt)
    arr = ord_v(arr)

    # Transform into binary, concatenate strings and back to array
    # Also inserts EOM
    arr = bin_v(arr, width=8)
    b_str = ''.join(arr) + '11111110'
    arr = np.array(list(b_str)).astype('int8')
    
    return arr

# Hides message inside image
def steganography_encode(img, txt, bit_plane):
    
    # Flatten image
    shape = img.shape
    out = img.ravel()
    
    # Get needed amount of pixels
    out_txt = out[:txt.size]
    
    # Bitwise operations
    mask = np.left_shift(1, bit_plane)
    txt_plane = np.left_shift(txt, bit_plane)
    txt_plane = np.where(txt == 1, txt_plane, np.invert(mask))
    out[:txt.size] = np.where(txt == 1, np.bitwise_or(txt_plane, out_txt), np.bitwise_and(txt_plane, out_txt))
    
    # Reshape and return
    return out.reshape(shape)

# Self-explanatory
def show_image(img, name='Image'):
    imshow(name, img)
    waitKey(0)
    destroyAllWindows()

def save_image(img, name, folder):
    imwrite(join(folder, name), img)
    
if __name__ == "__main__":
    main()

