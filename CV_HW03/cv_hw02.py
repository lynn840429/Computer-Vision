#****************************************************************************
#  FileName     [ cv_hw03.py ]
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


# (a) original image and its histogram
def Histogram(content_np, width, height, config):
	histogram_np = np.zeros(shape=(256))

	for x in range(width):
		for y in range(height):
			histogram_np[content_np[x][y]] += 1

	plt.figure(0)
	plt.bar(np.arange(256), histogram_np, 1)
	plt.title('Lena Histogram')
	# plt.xlabel("bpp value")
	# plt.ylabel("bpp number")
	plt.savefig(config.his)
	plt. close(0)
	print("Histogram Image Finish!")

# (b) image with intensity divided by 3 and its histogram
def Histogram_3(content_np, width, height, config):
	histogram_np = np.zeros(shape=(256))
	content_np = content_np // 3

	for x in range(width):
		for y in range(height):
			histogram_np[content_np[x][y]] += 1

	plt.figure(0)
	plt.bar(np.arange(256), histogram_np, 1)
	plt.title('Lena Histogram - Intensity Divided by 3')
	# plt.xlabel("bpp value")
	# plt.ylabel("bpp number")
	plt.savefig(config.his3)
	plt. close(0)
	Image.fromarray(np.uint8(content_np)).save(config.lena_his3)
	print("Histogram_3 Image Finish!")

# (c) image after applying histogram equalization to (b) and its histogram
def Equalization(content_np, width, height, config):
	histogram_np = np.zeros(shape=(256))
	content_np = cv2.equalizeHist(content_np//3)

	for x in range(width):
		for y in range(height):
			histogram_np[content_np[x][y]] += 1

	plt.figure(0)
	plt.bar(np.arange(256), histogram_np, 1)
	plt.title('Lena Histogram - Histogram Equalization')
	# plt.xlabel("bpp value")
	# plt.ylabel("bpp number")
	plt.savefig(config.equ)
	plt. close(0)
	Image.fromarray(np.uint8(content_np)).save(config.lena_equ)
	print("Equalization Image Finish!")


def main(config):
	# Image Pre-processing
	content = load_image(config.init_pict)
	width, height = content.size
	print("Image width=", width, ", Image height=", height, "\n")
	content_np = np.asarray(content).copy()

	# (a) original image and its histogram
	Histogram(content_np, width, height, config)
	# (b) image with intensity divided by 3 and its histogram
	Histogram_3(content_np, width, height, config)
	# (c) image after applying histogram equalization to (b) and its histogram
	Equalization(content_np, width, height, config)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--init_pict', type=str, default='lena.bmp')

	# (a) original image and its histogram
	parser.add_argument('--his', type=str, default='histogram lena.jpg')
	# (b) image with intensity divided by 3 and its histogram
	parser.add_argument('--his3', type=str, default='histogram_3 lena.jpg')
	parser.add_argument('--lena_his3', type=str, default='lena_h3 lena.jpg')
	# (c) image after applying histogram equalization to (b) and its histogram
	parser.add_argument('--equ', type=str, default='histogram_equ lena.jpg')
	parser.add_argument('--lena_equ', type=str, default='lena_equ lena.jpg')

	config = parser.parse_args()
	main(config)
