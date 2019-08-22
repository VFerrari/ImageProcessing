import cv2
import numpy as np

def negative(filename):
	img = 255 - cv2.imread(filename, 0)
	cv2.imwrite('Outputs/negative.png', img)

def intensity(filename):
	img = cv2.imread(filename, 0)
	img = np.uint8(((img/255) * 100) + 100)
	cv2.imwrite('Outputs/intensity.png', img)

def transform_intensity(filename):
	img = 255 - cv2.imread(filename, 0)
	img = np.uint8(((img/255) * 100) + 100)
	cv2.imwrite('Outputs/transform_intensity.png', img)
