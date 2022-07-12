#****************************************************************************
#  FileName     [ cv_hw06.py ]
#  Author       [ Lynn ]
#****************************************************************************
import os
import argparse
import numpy as np
import cv2

from PIL import Image

# Binarize image (threshold at 128)
def Binarize_cv2(config, img):
	ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
	cv2.imwrite(config.bin, thresh)
	print("Binarize Lena Done")
	return thresh

# Downsampled image
def downsampled(config, img, width, height):
	Img_sampled = np.zeros((width//8, height//8), np.uint8)
	
	for w in range(0, width, 8):
		for h in range(0, height, 8):
			Img_sampled[w//8][h//8] = img[w][h][0]

	cv2.imwrite(config.ds, Img_sampled)
	print("Downsampled Lena Done")
	return Img_sampled

## 8-connected neighborhood
## x7|x2|x6
## --------
## x3|x0|x1
## --------
## x8|x4|x5 
# h Function for Yokoi
def hFunc_4(b ,c, d, e):
	if (b==c and (d!=b or e!=b)):
		return 'q'
	elif (b==c and (d==b and e==b)):
		return 'r'
	elif (b!=c):
		return 's'

# f Function for Yokoi
def fFunc_4(a1, a2, a3, a4):
	if ([a1, a2, a3, a4].count('r')==4): 
		return 5
	else:
		return [a1, a2, a3, a4].count('q')

# Yokoi Connectivity Number
def Yokoi(config, img):
	width, height = img.shape
	window_8 = np.zeros((3, 3), np.uint8)
	yokoi_np = np.zeros((width, height), np.uint8)

	# Expand Img
	img_add4 = np.zeros((width+2, height+2), np.uint8)
	img_add4[1:width+1, 1:height+1] = img

	for w in range(1, width+1, 1):
		for h in range(1, height+1, 1):
			if (img_add4[w][h]==255):
				window_8 = img_add4[w-1:w+2, h-1:h+2]

				yokoi_np[w-1][h-1] = fFunc_4( 													 \
						hFunc_4(window_8[1][1], window_8[1][2], window_8[0][2], window_8[0][1]), \
						hFunc_4(window_8[1][1], window_8[0][1], window_8[0][0], window_8[1][0]), \
						hFunc_4(window_8[1][1], window_8[1][0], window_8[2][0], window_8[2][1]), \
						hFunc_4(window_8[1][1], window_8[2][1], window_8[2][2], window_8[1][2]))

	print("Yokoi Connectivity Number Lena Done")
	return yokoi_np

# Write txt
def write_txt(config, img):
	y_w, y_h = img.shape
	file = open("Yokoi.txt", "w")
	for w in range(y_w):
		for h in range(y_h):
			if (img[w][h]=='0'):
				file.write(' ')
			else:
				file.write(img[w][h])
		file.write('\n')
	print("Write txt Done")


def main(config):
	# Read in Image
	img = cv2.imread(config.init_pict)
	width, height, channel = img.shape
	print("Image width =", width, ", Image height =", height, "\n")

	# Binarize image (threshold at 128)
	Img_binarize = Binarize_cv2(config, img)

	# Downsampled image
	Img_downsampled = downsampled(config, Img_binarize, width, height)

	# Yokoi Connectivity Number
	Img_yokoi = Yokoi(config, Img_downsampled).astype(str)

	# Write txt
	write_txt(config, Img_yokoi)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--init_pict', type=str, default='lena.bmp')

	# Binarize image (threshold at 128)
	parser.add_argument('--bin', type=str, default='binarize_lena.bmp')
	# Downsampled image
	parser.add_argument('--ds', type=str, default='downsampled_lena.bmp')

	config = parser.parse_args()
	main(config)
