# Victor Ferreira Ferrari, RA 187890
# MC920 - Introduction to Image Digital Processing
# Project 4 - Steganography: Decoder
# Last modified: 25/10/2019

from cv2 import imwrite, imread
from cv2 import imshow, waitKey, destroyAllWindows, IMREAD_COLOR
from os.path import join
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Adds hidden message in an image.')
    parser.add_argument('img', help='Name of the file containing the coded image (PNG)')
    parser.add_argument('bit_plane', help='Bit plane in which to find the message.', type=int)
    parser.add_argument('out_txt', help='Name of the file to write the hidden text.')
    parser.add_argument('--folder', help='Folder to save coded image (defaults to Outputs/).', default='Outputs')
    args = parser.parse_args()

    # Read color image
    img = imread(args.img, IMREAD_COLOR)
    
    bin_seq = steganography_decode(img, args.bit_plane)
    txt = ascii_to_txt(bin_seq)
    
    # Saving txt
    out = join(args.folder, args.out_txt)
    with open(out, 'w') as f:
        n = f.write(txt)

# Decodes hidden message in image.
def steganography_decode(img, bit_plane):
    img = img.ravel()
    
    # Get message bits
    img = np.right_shift(img, bit_plane)
    bits = np.bitwise_and(img, 1)
    
    # Transform to integers
    arr_bits = np.array(np.array_split(bits, 8))
    msg = np.packbits(arr_bits)

    return msg

# Transforms binary sequence into a string.
def ascii_to_txt(bin_msg):
    
    # Getting EOM
    end = min(np.argwhere(bin_msg == 254)[0][0], len(bin_msg))
    actual_msg = bin_msg[:end]
    
    return actual_msg.tostring().decode('UTF-8')

if __name__ == "__main__":
    main()
