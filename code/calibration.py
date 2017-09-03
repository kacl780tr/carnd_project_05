import numpy as np
import cv2
import glob
import skimage.io as skio
import skimage.color as skcl
import skimage.util as skul
import pickle
import warnings


class Camera(object):
	"""
	Simple Camera object to manage matrix + distortion correction
	"""
	def __init__(self, matrix, distort):
		assert matrix is not None
		assert distort is not None
		self.__matrix = matrix
		self.__distort = distort
	
	def get_matrix(self):
		return self.__matrix

	def get_distortion(self):
		return self.__distort

	def apply_correction(self, image):
		return cv2.undistort(image, self.__matrix, self.__distort, newCameraMatrix=self.__matrix)


def calibration_data_extract(fileglob, gridsize=(9,6), display=False):
	"""
	Extract camera calibration data from a series of image files, assumed to be of a chessboard
	param: fileglob: the file globbing pattern used to search for the files
	param: gridsize: the size of the chessboard in each image
	param: display: whether to display the images (with corners) as they are processed (optional)
	return: a tuple (actual points, image points, image_shape)
	"""
	points_ref = np.zeros((gridsize[0]*gridsize[1], 3), np.float32)
	points_ref[:,:2] = np.mgrid[0:gridsize[0], 0:gridsize[1]].T.reshape(-1, 2) # create [x, y, z] coordinate array assuming regular spacing
	
	points_actual = []
	points_image = []
	image_shape = None

	calib_files = glob.glob(fileglob)
	for i, fn in enumerate(calib_files):
		img_colr = skio.imread(fn)
		img_gray = skcl.rgb2grey(img_colr)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			img_gray = skul.img_as_ubyte(img_gray)
		retval, corners = cv2.findChessboardCorners(img_gray, gridsize)
		if retval:
			if image_shape is None:
				image_shape = img_colr.shape[:2]			# we assume that all reference images are the same size, as defined by the camera
			points_image.append(corners)
			points_actual.append(points_ref)
			if display:
				cv2.drawChessboardCorners(image_colr, gridsize, corners, retval)
				cv2.imshow(fn, img_colr)
	if display:
		cv2.destroyAllWindows()
	return points_actual, points_image, image_shape


def calibration_compute(data_actual, data_image, image_size):
	"""
	Calculate camera calibration from data extracted from reference images
	param: data_actual: the actual reference points for the imaged objects
	param: data_image: the corresponding image points for the imaged objects
	param: image_size: the (x, y) size of the images from which the data were extracted
	return: Camera object if successful, None otherwise
	"""
	retval, cam_matrix, cam_distort, _, _ = cv2.calibrateCamera(data_actual, data_image, image_size, None, None)
	if retval:
		return Camera(cam_matrix, cam_distort)
	else:
		return None


def calibration_dump(filename, camera):
	"""
	Dump the camera calibration to the designated file
	param: filename: the desired dump file
	param: camera: the camera calibration object
	return: boolean True for success
	"""
	calib = {
		"matrix": camera.get_matrix(),
		"distort": camera.get_distortion()
	}
	try:
		with open(filename, "wb") as df:
			pickle.dump(calib, df)
			return True
	except Exception:
		return False


def calibration_load(filename):
	"""
	Load the camera calibration from the designated file
	param: filename
	return: Camera object if successful, None otherwise
	"""
	try:
		with open(filename, "rb") as lf:
			calib = pickle.load(lf)
			return Camera(calib["matrix"], calib["distort"])
	except Exception:
		return None


def calibration_retrieve(filename, fileglob=None):
	"""
	Retrieves the camera calibration if already computed, otherwise it is computed, saved, and returned if the calibration fileglob is provided
	param: filename: the name of the calibration dump file
	param: fileglob: the glob pattern of calibration images (optional)
	return: Camera object if successful, None otherwise
	note: if gridsize != (9,6) then calibration must be computed directly using the provided functions
	"""
	camera = calibration_load(filename)
	if camera is not None: return camera
	if fileglob is not None:
		actual, image, size = calibration_data_extract(fileglob)
		camera = calibration_compute(actual, image, size)
		if camera is not None:
			calibration_dump(filename, camera)
			return camera
	return None