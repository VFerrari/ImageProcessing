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
# Assumes square image (and possible to do a 4x4 mosaic), receives new order.
def mosaic(filename, arrange):
	img     = cv2.imread(filename, 0)
	arrange = np.array(arrange)
	p_len   = np.uint16(img.shape[0]/4)
	
	# Split image in 16 blocks
	i=0
	blocks = np.ndarray(16, np.ndarray)
	for y in range(0, img.shape[0], p_len):
		for x in range(0, img.shape[1], p_len):
			blocks[i] = img[y:y+p_len, x:x+p_len]
			i+=1
	
	# Concatenate rows of blocks, then all rows.
	rows = np.ndarray(4, np.ndarray)
	for i in range(4):
		rows[i] = np.concatenate(blocks[arrange[i]], axis=1)
	img = np.concatenate(rows)
	cv2.imwrite('Outputs/mosaic.png', img)

# Part 5
# Combine 2 images by weighted average
def combine(filename1, filename2, wei1, wei2):
	img1 = cv2.imread(filename1, 0)
	img2 = cv2.imread(filename2, 0)
	ans  = wei1*img1 + wei2*img2
	cv2.imwrite('Outputs/combine_' + str(wei1) + '_' + str(wei2) + '.png', ans)
