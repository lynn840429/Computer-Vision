#****************************************************************************
#  FileName     [ cv_hw02.py ]
#  Author       [ Lynn ]
#****************************************************************************
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.ndimage.measurements import label

# Image Pre-processing
def load_image(image_path):
	if not os.path.exists(image_path):
		print('Image not exit')
	else:
		image = Image.open(image_path)
		print('Input image:', image_path)
		return image

# (a) a binary image (threshold at 128)
def Binarize(content_np, content_np_new, width, height, config):
	for x in range(width):
		for y in range(height):
			if content_np[x][y]<128:
				content_np_new[x][y] = 0
			else:
				content_np_new[x][y] = 255
	Image.fromarray(np.uint8(content_np_new)).save(config.bin)
	print("Binary Image Finish!")

def Binarize_cv2(config):
	img = cv2.imread(config.init_pict)
	ret, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	cv2.imwrite("cv2_"+config.bin, thresh)
	print("Binary_cv2 Image Finish!")

# (b) a histogram
def Histogram(content_np, width, height, config):
	histogram_np = np.zeros(shape=(256))

	for x in range(width):
		for y in range(height):
			histogram_np[content_np[x][y]] += 1

	x = np.arange(256)
	w = 0.3
	plt.bar(x, histogram_np, w)
	plt.title('Lena Histogram')
	plt.xlabel("bpp value")
	plt.ylabel("bpp number")
	plt.savefig(config.his)
	#plt.show()
	print("Histogram Image Finish!")

def Histogram_cv2(config):
	img = cv2.imread(config.init_pict)
	vals = img.mean(axis=2).flatten()
	counts, bins = np.histogram(vals, range(257))
	plt.bar(bins[:-1]-0.5, counts, width=0.3)
	plt.title('Lena Histogram')
	plt.xlabel("bpp value")
	plt.ylabel("bpp number")
	plt.savefig("cv2_"+config.his)
	#plt.show()
	print("Histogram_cv2 Image Finish!")

# (c) connected components
def con_com(content_np_new, width, height, config):
	# Initial Binarize Img
	img = cv2.imread(config.init_pict)
	ret, img2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

	# Labeled Img
	filter9 = np.ones((3, 3), dtype=np.int) 									# Eight connected
	filter9[0][0], filter9[0][2], filter9[2][0], filter9[2][2] = 0, 0, 0, 0		# Four connected
	label_np, ncomponents = label(content_np_new, filter9)
	dig_num, counts = np.unique(label_np, return_counts=True)
	img_dict = dict(zip(dig_num, counts))
	label_np2 = label_np.copy()
	img_num = []

	# connected components more than 500 pixels
	for l in range(len(img_dict)):
		if (img_dict[l]>500):
			img_num.append(l)
	img_num.remove(0)

	rect_info = np.zeros(shape=(len(img_num),6), dtype=int) 	# x, y, w, h, cx, cy

	for p in range(len(img_num)):
		for x in range(width):
			for y in range(height):
				if label_np[x][y] == img_num[p]:
					label_np[x][y] = 255
				else:
					label_np[x][y] = 0 

		rect_info[p][:4] = cv2.boundingRect(label_np.astype('uint8'))	# rectangle
		centroid = cv2.moments(label_np.astype('uint8'))				# line
		rect_info[p][4] = int(centroid['m10']/centroid['m00'])
		rect_info[p][5] = int(centroid['m01']/centroid['m00'])
		label_np = label_np2.copy()

	for draw in range(len(rect_info)):
		x, y, w, h, cx, cy = rect_info[draw, :]
		label_np = cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 0, 255), 2) 		# rectangle
		label_np = cv2.line(img2, (cx-10, cy), (cx+10, cy), (255, 0, 0), 2)		# line
		label_np = cv2.line(img2, (cx, cy-10), (cx, cy+10), (255, 0, 0), 2) 	# line

	Image.fromarray(np.uint8(label_np)).save(config.con)
	print("Connected_Components Image Finish!")


def main(config):
	# Image Pre-processing
	content = load_image(config.init_pict)
	width, height = content.size
	print("Image width=", width, ", Image height=", height, "\n")
	content_np = np.asarray(content).copy()
	content_np_new = np.zeros(shape=(width,height))

	# (a) a binary image (threshold at 128)
	Binarize(content_np, content_np_new, width, height, config)
	Binarize_cv2(config)
	# (b) a histogram
	Histogram(content_np, width, height, config)
	Histogram_cv2(config)
	# (c) connected components
	con_com(content_np_new, width, height, config)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--init_pict', type=str, default='lena.bmp')

	# (a) a binary image (threshold at 128)
	parser.add_argument('--bin', type=str, default='binarize lena.bmp')
	# (b) a histogram
	parser.add_argument('--his', type=str, default='histogram lena.jpg')
	# (c) connected components
	parser.add_argument('--con', type=str, default='connected components lena.bmp')

	config = parser.parse_args()
	main(config)
