import argparse
import os
import numpy as np
from PIL import Image

def load_image(image_path):
	if not os.path.exists(image_path):
		print('Image path not exit')
	else:
		image = Image.open(image_path)
		print('Input image:', image_path)
		return image


def Up_side_Down(content_np, content_np_new, width, height, config):
	for x in range(width):
		for y in range(height):
			content_np_new[width-x-1][y] = content_np[x][y]
	Image.fromarray(np.uint8(content_np_new)).save(config.ud)


def Right_side_Left(content_np, content_np_new, width, height, config):
	for x in range(width):
		for y in range(height):
			content_np_new[x][height-y-1] = content_np[x][y]
	Image.fromarray(np.uint8(content_np_new)).save(config.rl)

def Diagonally_Mirrored(content_np, content_np_new, width, height, config):
	for x in range(width):
		for y in range(height):
			content_np_new[x][y] = content_np[y][x]
	Image.fromarray(np.uint8(content_np_new)).save(config.dia)


def Rotate_45(content, width, height, config):
	content.rotate(-45, expand=True).save(config.rot)


def Shrink_Half(content, width, height, config):
	content.resize((width//2, height//2)).save(config.shr)


def Binarize(content_np, width, height, config):
	for x in range(width):
		for y in range(height):
			if content_np[x][y]<128:
				content_np[x][y] = 0
			else:
				content_np[x][y] = 255
	Image.fromarray(np.uint8(content_np)).save(config.bin)



def main(config):
	content = load_image(config.init_pict)
	width, height = content.size
	print("Pic width=", width, ", Pic height=", height)
	content_np = np.asarray(content).copy()
	content_np_new = np.zeros(shape=(width,height))

	Up_side_Down(content_np, content_np_new, width, height, config)
	Right_side_Left(content_np, content_np_new, width, height, config)
	Diagonally_Mirrored(content_np, content_np_new, width, height, config)
	Rotate_45(content, width, height, config)
	Shrink_Half(content, width, height, config)
	Binarize(content_np, width, height, config)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--init_pict', type=str, default='lena.bmp')

	# Part1. Write a program to do the following requirement.
	parser.add_argument('--ud', type=str, default='upside-down lena.bmp')
	parser.add_argument('--rl', type=str, default='right-side-left lena.bmp')
	parser.add_argument('--dia', type=str, default='diagonally mirrored lena.bmp')

	# Part2. Write a program or use software to do the following requirement.
	parser.add_argument('--rot', type=str, default='rotate lena.bmp')
	parser.add_argument('--shr', type=str, default='shrink lena.bmp')
	parser.add_argument('--bin', type=str, default='binarize lena.bmp')

	config = parser.parse_args()
	main(config)
