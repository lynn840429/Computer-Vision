#****************************************************************************
#  FileName     [ cv_hw04.py ]
#  Author       [ Lynn ]
#****************************************************************************
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Image Pre-processing
def load_image(image_path):
	if not os.path.exists(image_path):
		print('Image not exit')
	else:
		image = Image.open(image_path)
		print('Input image:', image_path)
		return image

# (a) Dilation
def Dilation(thresh, oct_Kernel, config):
	lena_dilation = cv2.dilate(thresh, oct_Kernel)
	Image.fromarray(np.uint8(lena_dilation)).save(config.dilation)
	print("Dilation Done")

# (b) Erosion
def Erosion(thresh, oct_Kernel, config):
	lena_erosion = cv2.erode(thresh, oct_Kernel)
	Image.fromarray(np.uint8(lena_erosion)).save(config.erosion)
	print("Erosion Done")

# (c) Opening
def Opening(thresh, oct_Kernel, config):
	lena_opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, oct_Kernel)
	Image.fromarray(np.uint8(lena_opening)).save(config.opening)
	print("Opening Done")

# (d) Closing
def Closing(thresh, oct_Kernel, config):
	lena_closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, oct_Kernel)
	Image.fromarray(np.uint8(lena_closing)).save(config.closing)
	print("Closing Done")

# (e) Hit-and-miss transform
def HitandMiss(thresh, width, height, config):
	# Img A & inv_A
	A_J 	= np.zeros((width, height), np.uint8)
	invA_K 	= np.zeros((width, height), np.uint8)
	HandM	= np.zeros((width, height), np.uint8)

	# A_J
	for i in range(0, width-1):
		for j in range(1, height):
			if(thresh[i][j-1]==255 and thresh[i][j]==255 and thresh[i+1][j]==255):
				A_J[i][j]=1

	# invA_K
	for i in range(1, width):
		for j in range(0, height-1):
			if(thresh[i-1][j]==0 and thresh[i-1][j+1]==0 and thresh[i][j+1]==0):
				invA_K[i][j]=1

	# HandM
	for i in range(width):
		for j in range(height):
			if (A_J[i][j]==1 and invA_K[i][j]==1):
				HandM[i][j] = 255

	Image.fromarray(np.uint8(HandM)).save(config.ham)
	print("HitandMiss Done")



def main(config):
	# Image Pre-processing
	content = load_image(config.init_pict)
	width, height = content.size
	print("Image width=", width, ", Image height=", height)
	content_np = np.asarray(content).copy()
	ret, thresh = cv2.threshold(content_np, 127, 255, cv2.THRESH_BINARY)
	
	# octogonal 3-5-5-5-3 kernel
	oct_Kernel = np.ones((5, 5), np.uint8)
	oct_Kernel[0, 0], oct_Kernel[0, 4], oct_Kernel[4, 0], oct_Kernel[4, 4] = 0, 0, 0, 0
	print('')

	# Call Functions
	# (a) Dilation
	Dilation(thresh, oct_Kernel, config)
	# (b) Erosion
	Erosion(thresh, oct_Kernel, config)
	# (c) Opening
	Opening(thresh, oct_Kernel, config)
	# (d) Closing
	Closing(thresh, oct_Kernel, config)
	# (e) Hit-and-miss transform
	HitandMiss(thresh, width, height, config)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--init_pict', type=str, default='lena.bmp')

	# (a) Dilation
	parser.add_argument('--dilation', type=str, default='lena_dilation.jpg')
	# (b) Erosion
	parser.add_argument('--erosion', type=str, default='lena_erosion.jpg')
	# (c) Opening
	parser.add_argument('--opening', type=str, default='lena_opening.jpg')
	# (d) Closing
	parser.add_argument('--closing', type=str, default='lena_closing.jpg')
	# (e) Hit-and-miss transform
	parser.add_argument('--ham', type=str, default='lena_h&m.jpg')

	config = parser.parse_args()
	main(config)
