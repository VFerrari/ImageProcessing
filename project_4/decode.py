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
    
    # Arguments
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
    
    # Get message
    img = np.right_shift(img, bit_plane)
    bits = np.bitwise_and(img, 1)
    
    # Transform into an array of integers
    msg = np.zeros(len(bits)//8)
    for i in range(0, len(bits), 8):
        msg[i//8] = bits[i]*128 + bits[i+1]*64 + bits[i+2]*32 + bits[i+3]*16 + bits[i+4]*8 + bits[i+5]*4 + bits[i+6]*2 + bits[i+7]

    return msg.astype('int')

# Transforms binary sequence into a string.
def ascii_to_txt(bin_msg):
    chr_v = np.vectorize(chr)
    
    # Getting EOF
    end = min(np.where(bin_msg == 254)[0][0], len(bin_msg))
    actual_msg = bin_msg[:end]
    actual_msg = chr_v(actual_msg)
    
    return ''.join(actual_msg)

if __name__ == "__main__":
    main()
