import cv2
import numpy as np

# Part 1
# Image negative
def negative(filename):
	img = 255 - cv2.imread(filename, 0)
	cv2.imwrite('Outputs/negative.png', img)

# Converting intensity interval to [100,200]
def intensity(filename):
	img = cv2.imread(filename, 0)
	img = np.uint8(((img/255) * 100) + 100)
	cv2.imwrite('Outputs/intensity.png', img)

# Converting intensity interval to [100,200] of the negative.
def transform_intensity(filename):
	img = 255 - cv2.imread(filename, 0)
	img = np.uint8(((img/255) * 100) + 100)
	cv2.imwrite('Outputs/transform_intensity.png', img)

# Part 2
# Adjusting brightness given gamma.
def brightness(filename, gamma):
	img = cv2.imread(filename,0)/255
	img = img**(1/gamma)
	img = np.uint8(img*255)
	cv2.imwrite('Outputs/brightness_' + str(gamma) + '.png', img)

# Part 3
# Extract bit plane of a monochromatic image.
def bit_plane(filename, order):
	img   = cv2.imread(filename, 0)
	img = ((img >> order) % 2)*255
	cv2.imwrite('Outputs/bit_plane_' + str(order) + '.png', img)
	
# Part 4
# 4x4 mosaic from a monochromatic image
def mosaic(filename):
	img   = cv2.imread(filename, 0)
	p_len = np.uint8(img.shape[0]/4)
	b = []
	for y in range(4):
		for x in range(4):
			aux = img[y*p_len:(y+1)*p_len, x*p_len:(x+1)*p_len]
			b.append(aux)
	r0 = np.concatenate((b[5],b[10],b[12],b[2]), axis=1)
	r1 = np.concatenate((b[7],b[15],b[0],b[8]), axis=1)
	r2 = np.concatenate((b[11],b[13],b[1],b[6]), axis=1)
	r3 = np.concatenate((b[3],b[14],b[9],b[4]), axis=1)
	img2 = np.concatenate((r0,r1,r2,r3))
	cv2.imwrite('Outputs/mosaic.png', img2)

# Part 5
# Combine 2 images by weighted average
def combine(filename1, filename2, wei1, wei2):
	img1 = cv2.imread(filename1, 0)
	img2 = cv2.imread(filename2, 0)
	ans  = wei1*img1 + wei2*img2
	cv2.imwrite('Outputs/combine_' + str(wei1) + '_' + str(wei2) + '.png', ans)
