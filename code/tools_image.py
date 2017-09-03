import numpy as np
import cv2
import skimage.color as skcl
import warnings


def make_binary(source_gray, threshold=(0, 255)):
	"""
	Apply a simple binary filter to a single-channel image
	param: source_gray: a single-channel image (not necessarily grayscale)
	param: threshold: a tuple(lower, upper) of limits to apply
	return: binary image (1 only where limits satisfied)
	"""
	binary = np.zeros_like(source_gray)
	binary[(source_gray > threshold[0]) & (source_gray < threshold[1])] = 1.0
	return binary


def count_channel(source):
	"""
	Count the number of color channels in an image
	param: source: the source image
	return: the number of color channels
	"""
	shape = source.shape
	if len(shape) < 3 or shape[2] < 2:		# might have alpha channel
		return 1
	else:
		return shape[2]


def make_grayscale(source):
	"""
	Convert a multi-channel image to grayscale, single-channel images are returned unchanged
	param: source: source image to convert
	return: single-channel image
	"""
	shape = source.shape
	if len(shape) < 3 or shape[-1] < 2:
		return source					# already single channel
	else:
		#return skcl.rgb2grey(source)	# skimage returns a float image
		return cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
	

def make_gaussian_blur(source, kernelsize=3):
	"""
	Apply a Gaussian blur filter to an image
	param: source: source image
	param: kernelsize: the Gaussion filter size (optional, default = 3)
	return: filtered image
	"""
	return cv2.GaussianBlur(source, (kernelsize, kernelsize), 0)


def make_canny_transform(source_gray, threshold=(0, 255)):
	"""
	Apply the Canny transform to a grayscale image
	param: source_gray: the source grayscale image
	param: threshold: the (lower, upper) threshold limits
	return: the transformed image
	"""
	return cv2.Canny(source_gray, threshold[0], threshold[1])


def search_hough_lines(source_canny, delta_rho, delta_theta, threshold, min_length, max_gap):
	"""
	Seanch for Hough lines in a Canny-transformed image
	param: source_canny: Canny-transformed source image
	param: delta_rho: radial search resolution
	param: delta_theta: angular search resolution
	param: threshold: the lower threshold for line occurrences
	param: min_length: the minimum line length
	param: max_gap: maximum gap between linkable lines
	return: array of lines in the form [x_begin, y_begin, x_end, y_end]
	"""
	lines = cv2.HoughLinesP(source_canny, delta_rho, delta_theta, threshold, np.array([]), minLineLength=min_length, maxLineGap=max_gap)
	return lines


def apply_region_mask(source, region):
	"""
	Apply a mask to an image such that only the portion defined by the region is retained
	param: source: the source image
	param: region: a set of vertices defining the region to be retained
	return: the masked image
	"""
	mask = np.zeros_like(source)
	if len(source.shape) > 2:
		channel_count = source.shape[2]
		mask_ignore = (255,) * channel_count
	else:
		mask_ignore = 255
	cv2.fillPoly(mask, region, mask_ignore)
	source_masked = cv2.bitwise_and(source, mask)
	return source_masked


def make_sobel_directed(source_gray, direction="x", kernel=3, threshold=(0, 255)):
	"""
	Apply a directed Sobel operator to a single-channel image
	param: source_gray: single-channel source image
	param: direction: "x" or "y" to select direction
	param: kernel: the Sobel kernel size (must be odd integer) (optional)
	param: threshold: the (lower, upper) threshold limits for binary operation
	return: binary image
	"""
	dx, dy = (1,0) if direction == "x" else (0,1)
	sobel = cv2.Sobel(source_gray, cv2.CV_64F, dx, dy, ksize=kernel)
	sobel_abs = np.abs(sobel)
	sobel_scaled = (255*sobel_abs/np.max(sobel_abs)).astype(np.uint8)
	sobel_binary = make_binary(sobel_scaled, threshold=threshold)
	return sobel_binary


def make_sobel_magnitude(source_gray, kernel=3, threshold=(0, 255)):
	"""
	Apply a magnitude Sobel operator to a single-channel image - x and y directions are combined
	param: source_gray: single-channel source image
	param: kernel: the Sobel kernel size (must be odd integer) (optional)
	param: threshold: the (lower, upper) threshold limits for binary operation
	return: binary image
	"""
	sobel_x = cv2.Sobel(source_gray, cv2.CV_64F, 1, 0, ksize=kernel)
	sobel_y = cv2.Sobel(source_gray, cv2.CV_64F, 0, 1, ksize=kernel)
	sobel_abs = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
	sobel_scaled = (255*sobel_abs/np.max(sobel_abs)).astype(np.uint8)
	sobel_binary = make_binary(sobel_scaled, threshold=threshold)
	return sobel_binary


def make_sobel_radial(source_gray, kernel=3, threshold=(0, np.pi/2)):
	"""
	Apply a radial Sobel operator to a single-channel image - x and y directions are combined to produce an angular estimate of the gradient
	param: source_gray: single-channel source image
	param: kernel: the Sobel kernel size (must be odd integer) (optional)
	param: threshold: the (lower, upper) threshold limits for binary operation
	return: binary image
	"""
	sobel_x = cv2.Sobel(source_gray, cv2.CV_64F, 1, 0, ksize=kernel)
	sobel_y = cv2.Sobel(source_gray, cv2.CV_64F, 0, 1, ksize=kernel)
	sobel_dir = np.arctan2(np.abs(sobel_y), np.abs(sobel_x))
	sobel_binary = make_binary(sobel_dir, threshold=threshold)
	return sobel_binary


def select_space_hls(source_rgb, space="S"):
	"""
	Convert an RGB image to HLS color space and select a specific channel
	param: source_rgb: RGB source image
	param: space: should be one of "S", "H", or "L" if not, "S" is assumed, or if a pair of channels is desired, in the form "X,Y"
	return: selected HLS channel(s)
	"""
	source_hls = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HLS)
	channels = ("".join(space.split()).split(","))
	if len(channels)  == 1:
		channel = 2
		if channels[0] == "H": channel = 0
		if channels[0] == "L": channel = 1
		return source_hls[:,:,channel]
	elif len(channels) == 2:
		channel = [0, 2]
		for i, chn in enumerate(channels):
			if chn == "H": channel[i] = 0
			if chn == "L": channel[i] = 1
			if chn == "S": channel[i] = 2
		return source_hls[:,:,channel[0]], source_hls[:,:,channel[1]]
	else:
		return source_hls


def select_space_hsv(source_rgb, space="S"):
	"""
	Convert an RGB image to HSV color space and select a specific channel
	param: source_rgb: RGB source image
	param: space: should be one of "S", "H", or "V" if not, "S" is assumed
	return: selected HSV channel
	"""
	source_hsv = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV)
	channel = 1
	if space == "H": channel = 0
	if space == "V": channel = 2
	return source_hsv[:,:,channel]


def draw_linepath(image, linepath, color=[255, 0, 0], linewidth=3):
	"""
	Draw a linepath onto an image in a continuous manner
	param: image: the image onto which the region should be drawn
	param: linepath represented as an [N x 2] array of [x, y] pairs
	param: color: the RGB color to be drawn
	param: linewidth: the width of the line drawn
	return: None
	"""
	for j in range(linepath.shape[0] - 1):
		start = tuple(linepath[j])
		end = tuple(linepath[j+1])
		cv2.line(image, start, end, color=color, thickness=linewidth)


def draw_textpath(image, textpath, position=(10, 50), color=[255, 190, 0], linewidth=1):
	"""
	Draw a textpath (string) onto an image centered at position
	param: image: the image onto which to draw the text
	param: textpath: the string to draw
	param: position: the starting position in pixels
	param: color: the desired color
	param: linewidth: the width of the text line drawn
	return: None
	"""
	cv2.putText(image, textpath, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=linewidth)


def make_overlay(base, overlay):
	"""
	Superimpose an overlay image onto a base image
	param: base: the base image
	param: overlay: the image to be overlain (must have same shape as base)
	return: the combined image
	"""
	result = cv2.addWeighted(base, 1, overlay, 0.3, 0)
	return result