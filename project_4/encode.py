# Victor Ferreira Ferrari, RA 187890
# MC920 - Introduction to Image Digital Processing
# Project 4 - Steganography: Encoder
# Last modified: 31/10/2019

from cv2 import imwrite, imread
from cv2 import imshow, waitKey, destroyAllWindows, IMREAD_COLOR
from os.path import join
import numpy as np
import argparse

def main():
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
    
    txt = txt_to_bits(txt)
    out = steganography_encode(img, txt, args.bit_plane)    
    
    show_image(out, name='Coded')
    save_image(out, args.out_img, args.folder)
    
    # Showing bit planes
    for i in range(3):
        bit = planes(out,i)
        for j in range(3):
            show_image(bit[:,:,j], name=('Bit Plane' + str(i) + ' Color' + str(j)))
            save_image(bit[:,:,j], name=('bit_plane_' + str(i) + '_' + str(j) + '_' + args.out_img), folder=args.folder)
    show_image(planes(out,7), name=('Bit plane' + str(7)))

# Transforms text into bit array
def txt_to_bits(txt):
    
    # Space at the end for EOM
    txt = txt + ' '

    # Transform into ascii codes and to bit array
    # Adds EOM = 254 
    asc = np.fromstring(txt, 'S1').view(np.uint8)
    asc[-1] = 254
    bits = np.unpackbits(asc)

    return bits

# Hides message inside image
def steganography_encode(img, txt, bit_plane):
    out = img.ravel()
    
    # Get needed amount of pixels
    # Bitwise operations: create mask, and invert if txt=0.
    # If mask, operation OR, if inverted, operation AND.
    out_txt= out[:txt.size]
    mask   = np.left_shift(1, bit_plane)
    n_mask = np.invert(mask)
    out[:txt.size] = np.where(txt, np.bitwise_or(mask, out_txt), np.bitwise_and(n_mask, out_txt))
    
    # Reshape and return
    return out.reshape(img.shape)

# Gets bit planes
def planes(img, order):
    return ((img >> order) % 2)*255

# Self-explanatory
def show_image(img, name='Image'):
    imshow(name, img)
    waitKey(0)
    destroyAllWindows()

def save_image(img, name, folder):
    imwrite(join(folder, name), img)
    
if __name__ == "__main__":
    main()
