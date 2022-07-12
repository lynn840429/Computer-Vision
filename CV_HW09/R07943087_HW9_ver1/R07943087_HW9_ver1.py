#****************************************************************************
#  FileName     [ cv_hw09.py ]
#  Author       [ Lynn ]
#****************************************************************************
import os
import argparse
import numpy as np
import math
import cv2

from PIL import Image

# Robert's Operator
def Robert(img, width, height, threshold):
	img_copy = img[:, :, 0]
	img_robert = np.ones((width, height), np.uint8)
	img_robert = img_robert*255
	mask1 = np.array([[-1, 0], 
					  [ 0, 1]])
	mask2 = np.array([[ 0,-1], 
				   	  [ 1, 0]])
	
	for w in range(width-1):
		for h in range(height-1):
			r1 = np.sum(mask1 * img_copy[w:w+2, h:h+2])
			r2 = np.sum(mask2 * img_copy[w:w+2, h:h+2])
			gm = math.sqrt(r1**2 + r2**2)
			if (gm>threshold):
				img_robert[w, h] = 0

	return img_robert
	
# Prewitt's Edge Detector
def Prewitt(img, width, height, threshold):
	img_copy = img[:, :, 0]
	img_prewitt = np.ones((width+2, height+2), np.uint8)
	img_prewitt = img_prewitt*255
	mask1 = np.array([[-1, -1, -1],
					  [ 0,  0,  0],
					  [ 1,  1,  1]])
	mask2 = np.array([[-1,  0,  1],
					  [-1,  0,  1],
					  [-1,  0,  1]])

	for w in range(1, width-1, 1):
		for h in range(1, height-1, 1):
			p1 = np.sum(mask1 * img_copy[w-1:w+2, h-1:h+2])
			p2 = np.sum(mask2 * img_copy[w-1:w+2, h-1:h+2])
			gm = math.sqrt(p1**2 + p2**2)
			if (gm>threshold):
				img_prewitt[w, h] = 0 

	return img_prewitt[1:width+1, 1:height+1]

# Sobel's Edge Detector
def Sobel(img, width, height, threshold):
	img_copy = img[:, :, 0]
	img_sobel = np.ones((width+2, height+2), np.uint8)
	img_sobel = img_sobel*255
	mask1 = np.array([[-1, -2, -1],
					  [ 0,  0,  0],
					  [ 1,  2,  1]])
	mask2 = np.array([[-1,  0,  1],
					  [-2,  0,  2],
					  [-1,  0,  1]])

	for w in range(1, width-1, 1):
		for h in range(1, height-1, 1):
			s1 = np.sum(mask1 * img_copy[w-1:w+2, h-1:h+2])
			s2 = np.sum(mask2 * img_copy[w-1:w+2, h-1:h+2])
			gm = math.sqrt(s1**2 + s2**2)
			if (gm>threshold):
				img_sobel[w, h] = 0 

	return img_sobel[1:width+1, 1:height+1]

# Frei and Chen's Gradient Operator
def Frei_Chen(img, width, height, threshold):
	img_copy = img[:, :, 0]
	img_frei_chen = np.ones((width+2, height+2), np.uint8)
	img_frei_chen = img_frei_chen*255
	r2 = math.sqrt(2)
	mask1 = np.array([[-1,-r2, -1],
					  [ 0,  0,  0],
					  [ 1, r2,  1]])
	mask2 = np.array([[-1,  0,  1],
					  [-r2, 0, r2],
					  [-1,  0,  1]])

	for w in range(1, width-1, 1):
		for h in range(1, height-1, 1):
			s1 = np.sum(mask1 * img_copy[w-1:w+2, h-1:h+2])
			s2 = np.sum(mask2 * img_copy[w-1:w+2, h-1:h+2])
			gm = math.sqrt(s1**2 + s2**2)
			if (gm>threshold):
				img_frei_chen[w, h] = 0 

	return img_frei_chen[1:width+1, 1:height+1]

# Kirsch's Compass Operator
def Kirsch(img, width, height, threshold):
	img_copy = img[:, :, 0]
	img_kirsch = np.ones((width+2, height+2), np.uint8)
	img_kirsch = img_kirsch*255
	Kir = np.array([ [[-3, -3,  5], [-3,  0,  5], [-3, -3,  5]],
					 [[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]],
					 [[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]],
					 [[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]],
					 [[ 5, -3, -3], [ 5,  0, -3], [ 5, -3, -3]],
					 [[-3, -3, -3], [ 5,  0, -3], [ 5,  5, -3]],
					 [[-3, -3, -3], [-3,  0, -3], [ 5,  5,  5]],
					 [[-3, -3, -3], [-3,  0,  5], [-3,  5,  5]] ])

	for w in range(1, width-1, 1):
		for h in range(1, height-1, 1):
			k_list = []
			for k in range(8):
				ksum = np.sum(Kir[k] * img_copy[w-1:w+2, h-1:h+2])
				k_list.append(ksum)
				gmax = max(k_list)
			if (gmax>threshold):
				img_kirsch[w, h] = 0 

	return img_kirsch[1:width+1, 1:height+1]					

# Robinson's Compass Operator
def Robinson(img, width, height, threshold):
	img_copy = img[:, :, 0]
	img_robinson = np.ones((width+2, height+2), np.uint8)
	img_robinson = img_robinson*255
	Rob = np.array([ [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
					 [[ 0, 1, 2], [-1, 0, 1], [-2,-1, 0]],
					 [[ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]],
					 [[ 2, 1, 0], [ 1, 0,-1], [ 0,-1,-2]],
					 [[ 1, 0,-1], [ 2, 0,-2], [ 1, 0,-1]],
					 [[ 0,-1,-2], [ 1, 0,-1], [ 2, 1, 0]],
					 [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]],
					 [[-2,-1, 0], [-1, 0, 1], [ 0, 1, 2]] ])

	for w in range(1, width-1, 1):
		for h in range(1, height-1, 1):
			r_list = []
			for r in range(8):
				rsum = np.sum(Rob[r] * img_copy[w-1:w+2, h-1:h+2])
				r_list.append(rsum)
				gmax = max(r_list)
			if (gmax>threshold):
				img_robinson[w, h] = 0 

	return img_robinson[1:width+1, 1:height+1]

# Nevatia-Babu 5x5 Operator
def Nevatia_Babu(img, width, height, threshold):
	img_copy = img[:, :, 0]
	img_nevatia_babu = np.ones((width+4, height+4), np.uint8)
	img_nevatia_babu = img_nevatia_babu*255
	NB_mask = np.array([ 
		[[ 100, 100, 100, 100, 100], [ 100, 100, 100, 100, 100], [   0,   0,   0,   0,   0], [-100,-100,-100,-100,-100], [-100,-100,-100,-100,-100]],
	    [[ 100, 100, 100, 100, 100], [ 100, 100, 100,  78, -32], [ 100,  92,   0, -92,-100], [  32, -78,-100,-100,-100], [-100,-100,-100,-100,-100]],
	    [[ 100, 100, 100,  32,-100], [ 100, 100,  92, -78,-100], [ 100, 100,   0,-100,-100], [ 100,  78, -92,-100,-100], [-100, -32,-100,-100,-100]],
	    [[-100,-100,   0, 100, 100], [-100,-100,   0, 100, 100], [-100,-100,   0, 100, 100], [-100,-100,   0, 100, 100], [-100,-100,   0, 100, 100]],
	    [[-100,  32, 100, 100, 100], [-100, -78,  92, 100, 100], [-100,-100,   0, 100, 100], [-100,-100, -92,  78, 100], [-100,-100,-100, -32, 100]],
	    [[ 100, 100, 100, 100, 100], [ -32,  78, 100, 100, 100], [-100, -92,   0,  92, 100], [-100,-100,-100, -78,  32], [-100,-100,-100,-100,-100]] ])

	for w in range(2, width-2, 1):
		for h in range(2, height-2, 1):
			nb_list = []
			for nb in range(6):
				nbsum = np.sum(NB_mask[nb] * img_copy[w-2:w+3, h-2:h+3])
				nb_list.append(nbsum)
				gmax = max(nb_list)
			if (gmax>threshold):
				img_nevatia_babu[w, h] = 0 

	return img_nevatia_babu[2:width+2, 2:height+2]


def main(Config):
	# Read in Image
	Img = cv2.imread(Config.init_pict)
	Width, Height, Channel = Img.shape
	print("Image width =", Width, ", Image height =", Height, "\n")

	# (a) Robert's Operator: 12
	Img_Robert = Robert(Img, Width, Height, 12)
	cv2.imwrite("Lena_Robert_12.bmp", Img_Robert)
	print("Lena_Robert_12.bmp")
	# (b) Prewitt's Edge Detector: 24
	Img_Prewitt = Prewitt(Img, Width, Height, 24)
	cv2.imwrite("Lena_Prewitt_24.bmp", Img_Prewitt)
	print("Lena_Prewitt_24.bmp")
	# (c) Sobel's Edge Detector: 38
	Img_Sobel = Sobel(Img, Width, Height, 38)
	cv2.imwrite("Lena_Sobel_38.bmp", Img_Sobel)
	print("Lena_Sobel_38.bmp")
	# (d) Frei and Chen's Gradient Operator: 30
	Img_Frei_Chen = Frei_Chen(Img, Width, Height, 30)
	cv2.imwrite("Lena_Frei_Chen_30.bmp", Img_Frei_Chen)
	print("Lena_Frei_Chen_30.bmp")
	# (e) Kirsch's Compass Operator: 135
	Img_Kirsch = Kirsch(Img, Width, Height, 135)
	cv2.imwrite("Lena_Kirsch_135.bmp", Img_Kirsch)
	print("Lena_Kirsch_135.bmp")
	# (f) Robinson's Compass Operator: 43
	Img_Robinson = Robinson(Img, Width, Height, 43)
	cv2.imwrite("Lena_Robinson_43.bmp", Img_Robinson)
	print("Lena_Robinson_43.bmp")
	# (g) Nevatia-Babu 5x5 Operator: 12500
	Img_Nevatia_Babu = Nevatia_Babu(Img, Width, Height, 12500)
	cv2.imwrite("Lena_Nevatia-Babu_12500.bmp", Img_Nevatia_Babu)
	print("Lena_Nevatia-Babu_12500.bmp")


if __name__ == "__main__":
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--init_pict', type=str, default='lena.bmp')
	Config = Parser.parse_args()

	main(Config)
