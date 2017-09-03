import numpy as np
import cv2


class Perspective(object):
	"""
	Simple Perspective object to manage forward and reverse perspective transformations
	"""
	def __init__(self, source, target):
		assert source is not None
		assert target is not None
		try:
			self.__forward = make_perspective(source, target)
			self.__reverse = make_perspective(target, source)
		except Exception:
			self.__forward = None
			self.__reverse = None

	def get_forward(self):
		return self.__forward

	def get_reverse(self):
		return self.__reverse

	def apply(self, image):
		return apply_perspective(image, self.__forward)

	def unapply(self, image):
		return apply_perspective(image, self.__reverse)

	def point_apply(self, array):
		return apply_perspective_point(array, self.__forward)

	def point_unapply(self, array):
		return apply_perspective_point(array, self.__reverse)


def make_perspective(source, target):
	return cv2.getPerspectiveTransform(source, target)


def apply_perspective(image, transform):
	img_shape = (image.shape[1], image.shape[0])
	return cv2.warpPerspective(image, transform, img_shape, flags=cv2.INTER_LINEAR)


def apply_perspective_point(pointarray, transform):
	return cv2.perspectiveTransform(pointarray, transform)