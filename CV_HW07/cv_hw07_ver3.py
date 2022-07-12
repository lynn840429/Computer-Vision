#****************************************************************************
#  FileName     [ cv_hw07.py ]
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
	return thresh

# Downsampled image
def downsampled(config, img, width, height):
	Img_sampled = np.zeros((width//8, height//8), np.uint8)
	
	for w in range(0, width, 8):
		for h in range(0, height, 8):
			Img_sampled[w//8][h//8] = img[w][h][0]

	cv2.imwrite(config.ds, Img_sampled)
	return Img_sampled

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

				# x7|x2|x6
				# --------
				# x3|x0|x1
				# --------
				# x8|x4|x5
				yokoi_np[w-1][h-1] = fFunc_4( 													 \
						hFunc_4(window_8[1][1], window_8[1][2], window_8[0][2], window_8[0][1]), \
						hFunc_4(window_8[1][1], window_8[0][1], window_8[0][0], window_8[1][0]), \
						hFunc_4(window_8[1][1], window_8[1][0], window_8[2][0], window_8[2][1]), \
						hFunc_4(window_8[1][1], window_8[2][1], window_8[2][2], window_8[1][2]))

	return yokoi_np

# Pair Relationship Operator
def pair_relation(config, img):
	width, height = img.shape
	img_pair = np.zeros((width, height), np.uint8)
	img_ext = np.zeros((width+2, height+2), np.uint8)

	for w in range(width):
		for h in range(height):
			if (img[w][h]==1):
				img_pair[w][h] = 1
			elif (img[w][h]==0):
				img_pair[w][h] = 0
			else:
				img_pair[w][h] = 2

	img_ext[1:width+1, 1:height+1] = img_pair

	for w in range(1, width+1, 1):
		for h in range(1, height+1, 1):
			if (img_ext[w][h]!=0):
				if (img_ext[w][h]==0):
					img_pair[w-1][h-1] = 0
				elif (img_ext[w][h]==2):
					img_pair[w-1][h-1] = 2
				else:
					if (img_ext[w-1][h]==1 or img_ext[w][h-1]==1 or \
						img_ext[w+1][h]==1 or img_ext[w][h+1]==1):
						img_pair[w-1][h-1] = 1 	#p
					else:
						img_pair[w-1][h-1] = 2 	#q

	return img_pair

# h Function for Connected Shrink Operator
def hFunc_CSO(b ,c, d, e):
	if (b==c and (d!=b or e!=b)):
		return 1
	else:
		return 0

# Connected Shrink Operator
def connect_shrink(config, img_downsampled, img_pair):
	width, height = img_downsampled.shape
	window_8 = np.zeros((3, 3), np.uint8)
	img_ext = np.zeros((width+2, height+2), np.uint8)
	stop_flag = 0
	count_iter = 0

	while (stop_flag==0):
		img_yokoi = Yokoi(config, img_downsampled)
		img_pair = pair_relation(config, img_yokoi)
		img_ext[1:width+1, 1:height+1] = img_downsampled
		stop_flag = 1
		count_iter += 1

		for w in range(1, width+1, 1):
			for h in range(1, height+1, 1):
				if(img_ext[w][h]==255 and img_pair[w-1][h-1]==1):
					window_8 = img_ext[w-1:w+2, h-1:h+2]
					a1 = hFunc_CSO(window_8[1][1], window_8[1][2], window_8[0][2], window_8[0][1])
					a2 = hFunc_CSO(window_8[1][1], window_8[0][1], window_8[0][0], window_8[1][0])
					a3 = hFunc_CSO(window_8[1][1], window_8[1][0], window_8[2][0], window_8[2][1])
					a4 = hFunc_CSO(window_8[1][1], window_8[2][1], window_8[2][2], window_8[1][2])
					sum_a = a1 + a2 + a3 + a4
					x0 = 0

					if (sum_a!=1):
						x0 = 1

					if (x0==0):
						img_downsampled[w-1][h-1] = 0
						img_ext[w][h] = 0
						stop_flag = 0

		cv2.imwrite("thinning_"+str(count_iter)+".bmp", img_ext[1:width+1, 1:height+1])
	cv2.imwrite(config.thin, img_ext[1:width+1, 1:height+1])
	print("Thinning Lena Done")
	return img_ext[1:width+1, 1:height+1]

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
	Img_yokoi = Yokoi(config, Img_downsampled)
	# Pair Relationship Operator
	Img_pair = pair_relation(config, Img_yokoi)
	# Connected Shrink Operator
	Img_cso = connect_shrink(config, Img_downsampled, Img_pair).astype(str)
	# Write txt, func().astype(str)
	write_txt(config, Img_cso)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--init_pict', type=str, default='lena.bmp')

	# Binarize image (threshold at 128)
	parser.add_argument('--bin', type=str, default='binarize_lena.bmp')
	# Downsampled image
	parser.add_argument('--ds', type=str, default='downsampled_lena.bmp')
	# Connected Shrink Operator
	parser.add_argument('--thin', type=str, default='thinning_lena.bmp')

	config = parser.parse_args()
	main(config)
