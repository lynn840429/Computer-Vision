#****************************************************************************
#  FileName     [ cv_hw05.py ]
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
def Dilation(content_np, oct_Kernel, config):
	lena_dilation = cv2.dilate(content_np, oct_Kernel)
	Image.fromarray(np.uint8(lena_dilation)).save(config.dilation)
	print("Dilation Done")

# (b) Erosion
def Erosion(content_np, oct_Kernel, config):
	lena_erosion = cv2.erode(content_np, oct_Kernel)
	Image.fromarray(np.uint8(lena_erosion)).save(config.erosion)
	print("Erosion Done")

# (c) Opening
def Opening(content_np, oct_Kernel, config):
	lena_opening = cv2.morphologyEx(content_np, cv2.MORPH_OPEN, oct_Kernel)
	Image.fromarray(np.uint8(lena_opening)).save(config.opening)
	print("Opening Done")

# (d) Closing
def Closing(content_np, oct_Kernel, config):
	lena_closing = cv2.morphologyEx(content_np, cv2.MORPH_CLOSE, oct_Kernel)
	Image.fromarray(np.uint8(lena_closing)).save(config.closing)
	print("Closing Done")



def main(config):
	# Image Pre-processing
	content = load_image(config.init_pict)
	width, height = content.size
	print("Image width=", width, ", Image height=", height)
	content_np = np.asarray(content).copy()
	
	# octogonal 3-5-5-5-3 kernel
	oct_Kernel = np.ones((5, 5), np.uint8)
	oct_Kernel[0, 0], oct_Kernel[0, 4], oct_Kernel[4, 0], oct_Kernel[4, 4] = 0, 0, 0, 0
	print('')

	# Call Functions
	# (a) Dilation
	Dilation(content_np, oct_Kernel, config)
	# (b) Erosion
	Erosion(content_np, oct_Kernel, config)
	# (c) Opening
	Opening(content_np, oct_Kernel, config)
	# (d) Closing
	Closing(content_np, oct_Kernel, config)


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

	config = parser.parse_args()
	main(config)
