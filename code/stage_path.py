import numpy as np
import cv2


class PathFunction(object):
	"""
	Object to manage the paths traced by the lane lines
	"""
	def __init__(self, poly_lhs, poly_rhs, shape, n_grid=10, grid=None, realspace=False):
		assert poly_lhs is not None
		assert poly_rhs is not None
		self.__poly_lhs = poly_lhs
		self.__poly_rhs = poly_rhs
		self.__shape = shape
		if grid is not None: self.__grid = grid
		else: self.__grid = np.linspace(0, shape[0], n_grid)		# setup a standard grid to more easily allow operations
		self.__real_space = realspace

	def __curvature(self, poly):
		dx_dy = poly.deriv(m=1)										# compute curvature by averaging along path
		dx2dy = poly.deriv(m=2)
		den = np.abs(dx2dy(self.__grid))
		num = np.square(dx_dy(self.__grid))
		num += 1
		num = np.power(num, 3.0/2.0)
		num /= den
		return np.mean(num)

	def __real_ratios(self):
		yp = self.__shape[0]										# use fixed ratios for now
		f_x = 3.7/(self.__poly_rhs(yp) - self.__poly_lhs(yp))		# lane width as measured in pixels/3.7 meters
		f_y = 30.0/yp												# image height/30 meters (pretty rough estimate)
		return f_x, f_y

	def is_pixelspace(self):
		return not self.__real_space

	def get_lhs(self):
		return self.__poly_lhs

	def get_rhs(self):
		return self.__poly_rhs

	def get_lambda_lhs(self, offset=0):
		def _anchor_lhs(y):
			return (self.__poly_lhs(y) + offset)
		return _anchor_lhs

	def get_lambda_rhs(self, offset=0):
		def _anchor_rhs(y):
			return (self.__poly_rhs(y) + offset)
		return _anchor_rhs

	def get_grid(self):
		return self.__grid

	def curvature(self):
		return 0.5*(self.__curvature(self.__poly_lhs) + self.__curvature(self.__poly_rhs))

	def realspace_path(self):
		"""
		Convert the function to realspace by converting pixels to meters
		"""
		if self.__real_space: return self
		f_x, f_y = self.__real_ratios()
		vx_lhs = self.__poly_lhs(self.__grid)
		vx_lhs *= f_x;
		vx_rhs = self.__poly_rhs(self.__grid)
		vx_rhs *= f_x
		vy = np.copy(self.__grid)
		vy *= f_y
		act_shape = (f_y*self.__shape[0], f_x*self.__shape[1])
		act_lhs = np.polyfit(vy, vx_lhs, self.__poly_lhs.order)
		act_rhs = np.polyfit(vy, vx_rhs, self.__poly_rhs.order)
		return PathFunction(np.poly1d(act_lhs), np.poly1d(act_rhs), act_shape, grid=vy, realspace=True)

	def deviation(self):
		"""
		Calculate the deviation of vehicle centreline from lane centreline
		return: tuple(deviation in pixels, meters)
		"""
		if self.__real_space: return None
		f_x, _ = self.__real_ratios()
		yp = self.__shape[0]
		center_lane_px = 0.5*(self.__poly_lhs(yp) + self.__poly_rhs(yp))
		center_car_px = self.__shape[1]/2
		dev_px = center_car_px - center_lane_px
		dev_m = dev_px*f_x
		return (dev_px, dev_m)

	def draw(self, color=[0, 255, 0]):
		"""
		Allocate an RGB image array and draw the area between the left and right linepaths in the desired color
		param: color: the color to draw
		"""
		assert self.__real_space == False
		sx = self.__shape[1]
		sy = self.__shape[0]
		points_lhs = np.array([np.transpose(np.vstack([self.__poly_lhs(self.__grid), self.__grid]))]).astype(np.int32)
		points_rhs = np.array([np.transpose(np.vstack([self.__poly_rhs(self.__grid), self.__grid]))]).astype(np.int32)
		points_all = np.hstack([points_lhs, np.fliplr(points_rhs)])
		image = np.zeros((sy, sx, 3), dtype=np.uint8)
		cv2.fillPoly(image, points_all, color)
		return image


class PathFunctionBuilder(object):
	"""
	Object to manage the process of extracting lane line paths from an image
	"""
	def __init__(self, slice_fraction=0.10, kernel=50, margin=100, decay=0.25):
		self.slice_fraction = slice_fraction
		self.kernel = kernel
		self.margin = margin
		self.decay = decay
		self.previous_path = None

	def __split_point(self, shape, anchor):
		if self.previous_path is not None and self.previous_path.is_pixelspace():
			yp = shape[0]
			lhs = self.previous_path.get_lhs()(yp)
			rhs = self.previous_path.get_rhs()(yp)
			return int(0.5*(lhs + rhs))
		elif anchor is not None:
			return int(0.5*(anchor[0] + anchor[1]))
		else:
			return shape[1]//2
	
	def __make_anchor_lhs(self, anchor_points, offset=0):
		if self.previous_path is not None:
			return self.previous_path.get_lambda_lhs(offset), True		# return tuple(function, active)
		elif anchor_points is not None:
			return (anchor_points[0] + offset), False
		else:
			return None

	def __make_anchor_rhs(self, anchor_points, offset=0):
		if self.previous_path is not None:
			return self.previous_path.get_lambda_rhs(offset), True
		elif anchor_points is not None:
			return (anchor_points[1] + offset), False
		else:
			return None

	def __trace_path(self, img_split, anchor):
		slices, coords = make_slices(img_split, slice_frac=self.slice_fraction)
		path = make_linepath(slices, coords, anchor_function=anchor, kernel=self.kernel, margin=self.margin, decay=self.decay)
		return path

	def build_path(self, image, anchor_points=None):
		"""
		Build a path object from an image
		"""
		mid = self.__split_point(image.shape, anchor_points)
		image_lhs = image[:,:mid]
		image_rhs = image[:,mid:]
		path_lhs = self.__trace_path(image_lhs, self.__make_anchor_lhs(anchor_points))
		path_rhs = self.__trace_path(image_rhs, self.__make_anchor_rhs(anchor_points, offset=-mid))
		poly_lhs, _ = make_polynomial(path_lhs)															# use numpy poly1d objects
		poly_rhs, _ = make_polynomial(path_rhs, offset_x=mid)
		return PathFunction(poly_lhs, poly_rhs, image.shape)


def make_slices(image, slice_frac=0.10):
	"""
	Split an image into a number of horizontal slices
	param: image: the image to be sliced
	param: slice_frac: the fraction of image height represented by each slice (0, 1] (last slice may have different size)
	return: tuple([slices], [(ystart, yend)]) returned in order of increasing y coordinate
	"""
	assert slice_frac > 0.0 and slice_frac <= 1.0, "Invalid slice fraction"
	sx = image.shape[1]
	sy = image.shape[0]
	slices = []
	coords = []
	dy = slice_frac*sy
	ye, yb = 0,0
	i = -1
	while ye < sy:
		i += 1
		yb = int(i*dy)
		ye = min(int((i+1)*dy), sy)
		slc = np.sum(image[yb:ye,:], axis=0)
		slices.append(slc)
		coords.append((yb,ye))
	return slices, coords


def make_linepath(slices, coords, anchor_function=(None, False), kernel=50, margin=100, decay=0.01):
	"""
	Trace the path of a line vertically in a set of image slices
	param: slices: a list of image slices as 1d arrays (assumed in increasing y order)
	param: coords: a list of slice coordinates as 2-tuples
	param: anchor_function: a function returning estimated position as a function of y 
	param: kernel: the width of the convolution kernel to use
	param: margin: the allowed location change from slice to slice
	param: decay: the decay rate of the path mixture computation
	return: a list of (x, y) coordinate tuples
	"""
	assert len(slices) == len(coords)
	if anchor_function is None or anchor_function[1] == False:
		return make_linepath_static(slices, coords, anchor=anchor_function[0], kernel=kernel, margin=margin, decay=decay)
	else:
		return make_linepath_active(slices, coords, anchor_function=anchor_function[0], kernel=kernel, margin=margin, decay=decay)


def make_linepath_active(slices, coords, anchor_function=None, kernel=50, margin=100, decay=0.01):
	"""
	Trace the path of a line vertically in a set of image slices
	param: slices: a list of image slices as 1d arrays (assumed in increasing y order)
	param: coords: a list of slice coordinates as 2-tuples
	param: anchor_function: a function returning estimated position as a function of y 
	param: kernel: the width of the convolution kernel to use
	param: margin: the allowed location change from slice to slice
	param: decay: the decay rate of the path mixture computation
	return: a list of (x, y) coordinate tuples
	"""
	assert len(slices) == len(coords)
	path = []
	if len(slices) == 0: return path
	window = np.ones(kernel)
	N = slices[0].shape[0]
	indx = np.linspace(0, N - 1, N)
	indx2 = np.square(indx)
	decay_rate = -np.abs(decay)														# ensure decay < 0
	for slc, ypos in zip(reversed(slices), reversed(coords)):
		conv = np.convolve(window, slc, mode="same")
		conv /= (np.sum(conv) + np.spacing(1.0)) 
		yp = np.sum(ypos)//len(ypos)												# y coordinate of slice
		if anchor_function is not None:
			anchor_x = anchor_function(yp)
			x_min = int(max(anchor_x - kernel//2 - margin, 0))
			x_max = int(min(anchor_x + kernel//2 + margin, N))
			exp_x = np.dot(conv[x_min:x_max], indx[x_min:x_max])					# compute expected value of x
			if exp_x == 0: exp_x = anchor_x											# use current anchor point for empty slice
			else:
				exp_x2 = np.dot(conv[x_min:x_max], indx2[x_min:x_max])				# compute expected value of x^2
				var_x = exp_x2 - exp_x*exp_x										# compute variance
				sig_x = np.sqrt(np.abs(var_x))										# std deviation
				rate = decay_rate*sig_x/(exp_x + np.spacing(1.0))
				beta = np.exp(rate*np.abs(exp_x - anchor_x))						# mixing ratio
				exp_x = (1.0 - beta)*anchor_x + beta*exp_x							# new estimate is mixture of old and new
		else:
			exp_x = np.dot(conv, indx)
			if exp_x == 0: continue													# no signal for empty slice
		xp = exp_x
		path.append((xp, yp))
	return path


def make_linepath_static(slices, coords, anchor=None, kernel=50, margin=100, decay=0.01):
	"""
	Trace the path of a line vertically in a set of image slices
	param: slices: a list of image slices as 1d arrays (assumed in increasing y order)
	param: coords: a list of slice coordinates as 2-tuples
	param: anchor: an estimated initial position of the line 
	param: kernel: the width of the convolution kernel to use
	param: margin: the allowed location change from slice to slice
	param: decay: the decay rate of the path mixture computation
	return: a list of (x, y) coordinate tuples
	"""
	assert len(slices) == len(coords)
	path = []
	if len(slices) == 0: return path
	window = np.ones(kernel)
	N = slices[0].shape[0]
	indx = np.linspace(0, N - 1, N)
	indx2= np.square(indx)
	anchor_x = anchor
	decay_rate = -np.abs(decay)														# make sure decay is < 0
	for slc, ypos in zip(reversed(slices), reversed(coords)):
		conv = np.convolve(window, slc, mode="same")
		conv /= (np.sum(conv) + np.spacing(1.0))
		if anchor_x is not None:
			x_min = int(max(anchor_x - kernel//2 - margin, 0))
			x_max = int(min(anchor_x + kernel//2 + margin, N))
			exp_x = np.dot(conv[x_min:x_max], indx[x_min:x_max])					# compute expected value of x
			if exp_x == 0: exp_x = anchor_x											# use current anchor point for empty slice
			else:
				exp_x2 = np.dot(conv[x_min:x_max], indx2[x_min:x_max])				# compute expected value of x^2
				var_x = exp_x2 - exp_x*exp_x										# variance
				sig_x = np.sqrt(np.abs(var_x))										# std deviation
				rate = decay_rate*sig_x/(exp_x + np.spacing(1.0))
				beta = np.exp(rate*np.abs(exp_x - anchor_x))						# mixing ratio
				exp_x = (1.0 - beta)*anchor_x + beta*exp_x							# new estimate is mixture of old and new
				anchor_x = exp_x													# new anchor is current estimate
		else:
			exp_x = np.dot(conv, indx)
			if exp_x == 0: continue													# no signal for empty slice
		xp = exp_x
		yp = np.sum(ypos)//len(ypos)												# y coordinate of slice
		path.append((xp, yp))
	return path


def make_polynomial(path, order=2, offset_x=0):
	"""
	Fit a polynomial to the path traced by a line
	param: path: a list of (x, y) tuples describing the line
	param: order: the order of the polynomial to be fit (optional, default=2)
	param: offset_x: a fixed x offset to be applied to the x values
	return: tuple(poly1d object, y-grid) 
	"""
	n = len(path)
	vx = np.zeros((n), dtype=np.float32)
	vy = np.zeros((n), dtype=np.float32)
	for i, pt in enumerate(reversed(path)):
		vx[i] = pt[0] + offset_x
		vy[i] = pt[1]
	fit = np.polyfit(vy, vx, order)
	return np.poly1d(fit), vy

	