import numpy as np
import cv2
import display
import scipy.ndimage.measurements as sndm
from collections import deque


class Window(object):
	"""
	A simple window object representing a rectangular image region
	"""
	def __init__(self, x_lo, y_lo, x_hi, y_hi, phase=0):
		assert x_lo is not None, "invalid upper left coordinate"
		assert y_lo is not None, "invalid upper left coordinate"
		assert x_hi is not None, "invalid lower right coordinate"
		assert x_hi is not None, "invalid lower right coordinate"
		assert x_lo >= 0 and x_hi > x_lo, "invalid X-span"
		assert y_lo >= 0 and y_hi > y_lo, "invalid Y-span"
		self.__upper_left = (x_lo, y_lo)
		self.__lower_rght = (x_hi, y_hi)
		self.phase = phase

	def __clip_corner(self, shape, corner):
		"""
		Clip lower right to edge of image if necessary
		param: shape: the shape of the image
		return: clipped lower-right coordinates
		"""
		limit = shape[:2]
		return (min(shape[1], corner[0]), min(shape[0], corner[1]))

	def __make_span(self, ul, lr):
		return ((lr[0] - ul[0]), (lr[1] - ul[1]))

	def __make_intersection(self, this_ul, this_lr, that_ul, that_lr):
		int_ul = (max(this_ul[0], that_ul[0]), max(this_ul[1], that_ul[1]))
		int_lr = (min(this_lr[0], that_lr[0]), min(this_lr[1], that_lr[1]))
		return int_ul, int_lr

	def __make_union(self, this_ul, this_lr, that_ul, that_lr):
		uni_ul = (min(this_ul[0], that_ul[0]), min(this_ul[1], that_ul[1]))
		uni_lr = (max(this_lr[0], that_lr[0]), max(this_lr[1], that_lr[1]))
		return uni_ul, uni_lr

	def get_upper_left(self):
		"""
		return: upper left corner as tuple(x, y)
		"""
		return self.__upper_left

	def get_lower_right(self):
		"""
		return: lower right corner as tuple(x, y)
		"""
		return self.__lower_rght

	def get_span(self, shape=None):
		"""
		Compute the span in pixels
		param: shape: the image shape to which the window should be clipped (optional)
		return: a tuple(x-span, y-span) representing the (width, height) of the window
		"""
		if shape is None:
			return self.__make_span(self.__upper_left, self.__lower_rght)
		else:
			ul = self.__clip_corner(shape, self.__upper_left)
			lr = self.__clip_corner(shape, self.__lower_rght)
			return self.__make_span(ul, lr)

	def get_area(self, shape=None):
		span = self.get_span(shape)
		return span[0]*span[1]

	def draw(self, image, color=(0, 0, 255), width=1):
		"""
		Draw the window onto the image
		param: image: the image onto which the window will be drawn
		param: color: the color with which the window will be drawn (optional, default = blue)
		param: width: the linewidth to be used for drawing (optional, default = 1)
		return: self
		"""
		ul = self.__clip_corner(image.shape, self.__upper_left)
		lr = self.__clip_corner(image.shape, self.__lower_rght)
		cv2.rectangle(image, ul, lr, color, thickness=width)
		return self

	def extract(self, image):
		"""
		Extract the data corresponding to the window from the image
		param: image: the image from which the window will be extracted
		return: the data for the window sub-region of the image (we assume multi-channel image)
		"""
		ul = self.__clip_corner(image.shape, self.__upper_left)
		lr = self.__clip_corner(image.shape, self.__lower_rght)
		return image[ul[1]:lr[1],ul[0]:lr[0],:]

	def extract_scale(self, image, size=(64, 64)):
		"""
		Extract the data corresponding to the window and resize
		param: image: the image from which the window will be extracted
		param: size: a two-tuple representing the output size
		return: a sub-region of size=size of the data from the window sub-region
		"""
		return cv2.resize(self.extract(image), size, interpolation=cv2.INTER_NEAREST)

	def update_heatmap(self, map, step=1.0):
		"""
		Update a heatmap over the area represented by the window
		param: map: a 2D array containing the heatmap
		param: step: the step in heatmap value to apply (optional, default=1.0)
		return: self
		"""
		ul = self.__clip_corner(map.shape, self.__upper_left)
		lr = self.__clip_corner(map.shape, self.__lower_rght)
		map[ul[1]:lr[1],ul[0]:lr[0]] += step
		return self

	def normalized_coordinates(self, shape):
		"""
		Convert the window from pixels to normalized coordinates for an image with shape
		param: shape: the shape of the image for normalization
		return: a [1 x 4] array of float32 coordinates [x1, y1, x2, y2]
		"""
		norm_coords = np.zeros((1,4), dtype=np.float32)
		norm_coords[0] = max(float(self.__upper_left[0])/float(shape[1]), 0.0)
		norm_coords[1] = min(float(self.__upper_left[1])/float(shape[0]), 1.0)
		norm_coords[2] = max(float(self.__lower_rght[0])/float(shape[1]), 0.0)
		norm_coords[3] = min(float(self.__lower_rght[1])/float(shape[0]), 1.0)
		return norm_coords

	def shift(self, delta=(0,0), shape=None):
		"""
		Shift the window by delta pixels
		param: delta: 2-tuple of (dx, dy) in pixels
		param: shape: the boundary shape of the image (optional)
		return: a window object appropriately shifted 
		"""
		ul_n = (max(self.__upper_left[0] + delta[0], 0), max(self.__upper_left[1] + delta[1], 0))
		lr_n = (max(self.__lower_rght[0] + delta[0], 0), max(self.__lower_rght[1] + delta[1], 0))
		if shape is not None:
			ul_n = self.__clip_corner(map.shape, ul_n)
			lr_n = self.__clip_corner(map.shape, lr_n)
		return Window(ul_n[0], ul_n[1], lr_n[0], lr_n[1])

	def has_overlap(self, window, shape=None):
		"""
		Check if the window overlaps with another
		param: window: the window to test for overlap
		param: shape: the window shape for clipping (optional)
		return: boolean True or False
		"""
		if shape is not None:
			this_ul = self.__clip_corner(self.__upper_left, shape)
			this_lr = self.__clip_corner(self.__lower_rght, shape)
			that_ul = self.__clip_corner(window.get_upper_left(), shape)
			that_lr = self.__clip_corner(window.get_lower_right(), shape)
			int_ul, int_lr = self.__make_intersection(this_ul, this_lr, that_ul, that_lr)
		else:
			int_ul, int_lr = self.__make_intersection(self.__upper_left, self.__lower_rght, window.get_upper_left(), window.get_lower_right())
		int_span = self.__make_span(int_ul, int_lr)
		return bool(int_span[0] > 0 and int_span[1] > 0)


class WindowGroup(list):
	"""
	A class containing a group of window objects
	"""
	def __init__(self, base_size=(64, 64)):
		super().__init__()
		self.__baseline = base_size

	def draw(self, image, color=(0, 0, 255), width=1, indicator=None):
		"""
		Draw all windows in group onto the image
		param: image: the image onto which the windows will be drawn
		param: color: the color to use for drawing (optional)
		param: width: the line width to use for drawing (optional)
		param: indicator: an iterable that determines whether a given window (optional)
				should be included in the draw
		return: self
		"""
		if indicator is None:
			for wdw in self:
				wdw.draw(image, color=color, width=width)
		else:
			for v, wdw in zip(indicator, self):
				if v: wdw.draw(image, color=color, width=width)
		return self

	def extract_scale(self, image, size=None):
		"""
		Extract and scale image data from the windows
		param: image from which to extract data
		size: the output size as tuple(x, y) in pixels
		return: a 4D array of extracted image data shape = [window, size[0], size[1], image.shape[-1]]
		"""
		sz = self.__baseline if size is None else size
		sub_images = np.zeros((len(self), sz[0], sz[1], image.shape[-1]), dtype=image.dtype)
		for i, wdw in enumerate(self):
			sub_images[i,:,:,:] = wdw.extract_scale(image, size=sz)
		return sub_images

	def update_heatmap(self, map, step=1.0, indicator=None):
		"""
		Update a heatmap over the areas represented by windows
		param: map: a 2D array containing the heatmap
		param: step: the step in the heatmap value to apply (optional, default = 1.0)
		param: indicator: an iterable that determines whether a given window (optional)
				should be included in the heatmap
		"""
		if indicator is None:
			for wdw in self:
				wdw.update_heatmap(map, step=step)
			return self
		else:
			for v, wdw in zip(indicator, self):
				if v: wdw.update_heatmap(map, step=step)
		return self

	def select(self, indicator):
		"""
		Select a subset of windows
		param: indicator: an iterable that determines whether a given window
				should be included in the subset
		return: WindowGroup object containing subset
		"""
		subset = WindowGroup(self.__baseline)
		for i in range(len(indicator)):
			if indicator[i]:
				subset.append(self[i])
		return subset

	def normalized_coordinates(self, shape, indicator=None):
		"""
		Convert the windows into normalized coordinates for an image shape
		param: shape: the shape for normalization
		param: indicator: an iterable that determines whether an given window should be included (optional)
		return: an [N x 4] float array, where N is the number of windows
		"""
		if indicator is None:
			norm_coords = np.zeros((len(self), 4), dtype=np.float32)
			for i, wdw in enumerate(self):
				norm_coords[i,:] = wdw.normalized_coordinates(shape)
			return norm_coords
		else:
			coords = []
			for v, wdw in zip(indicator, self):
				if v: coords.append(wdw.normalized_coordinates(shape))
			return np.vstack(coords)

	def shift(self, delta=(0,0), shape=None, indicator=None):
		"""
		Create and return a WindowGroup containing shifted windows
		param: delta: 2-tuple of (dx, dy) in pixels
		param: shape: the boundary shape of the image (optional)
		param: indicator: an iterable that determines whether a given window is included (optional)
		return: a window object appropriately shifted 
		"""
		shifted = WindowGroup(self.__baseline)
		if indicator is None:
			for wdw in self:
				shifted.append(wdw.shift(delta, shape=shape))
		else:
			for v, wdw in zip(indicator, self):
				if v: shifted.append(wdw.shift(delta, shape=shape))
		return shifted

	def has_overlap(self, window, shape=None, indicator=None):
		"""
		Test all contained windows for overlap with given window
		param: window: the window to be tested for overlap
		param: shape: the boundary shape of the image (optional)
		param: indicator: an iterable that determines whether a given window is tested (optional)
		return: boolean True or False
		"""
		if indicator is None:
			for wdw in self:
				if wdw.has_overlap(window, shape=shape): return True
		else:
			for v, wdw in zip(indicator, self):
				if v:
					if wdw.has_overlap(window, shape=shape):
						return True
		return False

	def sort(self, key=None, reverse=False):
		"""
		Sort the windows according to a key
		param: key: the sorting key to be used (optional, default = window area)
		param: reverse: whether to reverse the sorting direction (optional, default = False)
		return: self
		"""
		if key is None: key = lambda w: w.get_area()
		super().sort(key=key, reverse=reverse)
		return self


class WindowBuilder(object):
	"""
	A class designed to build search windows based on a sequence of rules
	"""
	def __init__(self, baseline=(64,64), rules=[(1.0, 0.0, 1.0)], overlap=(0.5, 0.5), x_stretch=1.0, ceiling=0):
		"""
		Store the set of rules by which windows will be constructed
		param: baseline: all window data extracted from images will be resized to this size
		param: rules: the sequence of rules used to construct windows list(tuple(scale, frac_top, frac_bottom))
		param: overlap: the amount by which the windows will overlap
		param: x_stretch: the factor by which the x-dimension will be multiplied
		param: ceiling: if > 0, represents a cap on the overall number of windows (optional = 0) window lists a pruned randomly if necessary
		"""
		super().__init__()
		assert len(overlap) == 2, "invalid overlap"
		assert overlap[0] >= 0 and overlap[0] < 1.0, "invalid x overlap"
		assert overlap[1] >= 0 and overlap[1] < 1.0, "invalid y overlap"
		self.__base = baseline				# baseline window size = base output size for window data
		self.__rules = rules				# rule format = [(scale, frac_top, frac_bot),...]
		self.__overlap = overlap			# overlap = (overlap x, overlap y)
		self.__stretch = x_stretch			# stretch the width by x_stretch factor
		self.__ceiling = ceiling			# limit on the total number of boxes

	def __make_window(self, shape, factor):
		wx = self.__base[0]
		wy = self.__base[1]
		if factor > 0: 
			wx = int(factor*self.__stretch*self.__base[0])
			wy = int(factor*self.__base[1])
		return (min(wx, shape[1]), min(wy, shape[0]))

	def __make_limit(self, shape, range):
		limit_x = (0, shape[1])
		limit_y = (int(max(0, range[0]*shape[0])), int(min(shape[0], range[1]*shape[0])))
		return limit_x, limit_y

	def __make_windows(self, limit_x, limit_y, window, overlap):
		dx = int((1.0 - overlap[0])*window[0])
		bx = window[0] - dx
		nx = int((limit_x[1] - limit_x[0] - bx)/dx)
		dy = int((1.0 - overlap[1])*window[1])
		by = window[1] - dy
		ny = int((limit_y[1] - limit_y[0] - by)/dy)
		windows = []
		for j in range(ny):
			ys = j*dy + limit_y[0]
			ye = ys + window[1]
			for i in range(nx):
				xs = i*dx + limit_x[0]
				xe = xs + window[0]
				wndw = Window(xs, ys, xe, ye)
				windows.append(wndw)
		return windows

	def __make_all(self, shape):
		windows = WindowGroup(base_size=self.__base)
		for scl, r_t, r_b in self.__rules:
			window = self.__make_window(shape, scl)
			limit_x, limit_y = self.__make_limit(shape, (r_t, r_b))
			windows += self.__make_windows(limit_x, limit_y, window, self.__overlap)
		if self.__ceiling > 0 and len(windows) > self.__ceiling:
			while len(windows) > self.__ceiling:
				idx = np.random.randint(0, len(windows))
				windows.pop(idx)
		return windows

	def make_window_group(self, shape):
		"""
		Create a window group containing all windows from the window rules
		param: shape: the shape of the window over which to construct the window group
		return: WindowGroup object
		"""
		return self.__make_all(shape)


class HeatMap(object):
	"""
	A simple object representing a 2D heatmap
	"""
	def __init__(self, shape, threshold=1.0):
		self.__shape = shape[:2]
		self.__map = np.zeros(self.__shape)
		self.__scratch = np.zeros_like(self.__map)
		self.threshold = threshold

	def __apply_limit(self, limit):
		np.copyto(self.__scratch, self.__map)
		self.__scratch[self.__scratch < limit] = 0
		return self.__scratch

	def __get_threshold(self, threshold=None):
		lim = self.threshold if threshold is None else threshold
		return lim

	def merge(self, map, factor=1.0):
		"""
		Merge the current map with another
		param: map: the other map
		param: factor: the factor to be used for the sum (optional default = 1.0)
		return: reference to this map
		"""
		self.__map += (factor*map.get_map())
		return self

	def get_shape(self):
		"""
		param: None
		return: a tuple providing the shape of the map data
		"""
		return self.__shape

	def get_map(self):
		"""
		param: None
		return: reference to the map data
		"""
		return self.__map

	def get_level(self):
		"""
		param: None
		return: the maximum level present in the map
		"""
		return np.max(self.__map)

	def reset(self):
		"""
		Reset the map to zero value
		"""
		self.__map[:,:] = 0.0
		return self

	def make_binary(self):
		"""
		Convert map into a binary map where any value > 0 = 1.0
		"""
		self.__map[self.__map > 0] = 1.0
		return self

	def apply_threshold(self, threshold=None):
		lim = self.__get_threshold(threshold)
		self.__map[self.__map < lim] = 0.0
		return self

	def make_windows(self, threshold=None):
		"""
		Segment the heatmap into regions and create a WindowGroup containing
		param: threshold (optional default = value set at initialization)
		"""
		scratch = self.__apply_limit(self.__get_threshold(threshold))
		windows = WindowGroup()
		segmap, N = sndm.label(scratch)
		for lbl in range(1, N+1):
			where = (segmap == lbl).nonzero()
			where_x = np.array(where[1])
			where_y = np.array(where[0])
			try:
				wdw = Window(np.min(where_x), np.min(where_y), np.max(where_x), np.max(where_y))
				windows.append(wdw)
			except Exception:
				continue
		return windows


class HeatMapGroup(object):
	"""
	Object to allow sliding window of combined heatmaps
	"""
	def __init__(self, shape, count, limit_factor=0.5):
		self.__count = count
		self.__factor = limit_factor
		self.__queue = deque()
		self.__map = HeatMap(shape)
		self.__level = 0.0

	def reset(self):
		"""
		Reset the map group by clearing the queue and resetting the cumulative map
		param: None
		return: reference to this object
		"""
		self.__queue.clear()
		self.__map.reset()
		self.__level = 0.0
		return self

	def get_shape(self):
		"""
		Return a tuple giving the shape of the map
		param: None
		return: a 2-tuple (width, height)
		"""
		return self.__map.get_shape()

	def get_map(self):
		"""
		Return reference to cumulative map
		param: None
		return: reference to cumulative map
		"""
		return self.__map

	def add_map(self, map):
		"""
		Add a heatmap to the group
		param: map: the new heatmap
		return: an unused heatmap if available, else None
		"""
		old = None
		if len(self.__queue) >= self.__count:
			old = self.__queue.popleft()
			self.__map.merge(old, -1.0)							# subtract oldest map
			self.__level -= old.get_level()
		self.__map.merge(map.make_binary(), 1.0)				# add newest map
		self.__level += map.get_level()
		self.__queue.append(map)
		return old												# return disused heatmap for recycling

	def make_windows(self):
		"""
		Create a set of detected windows based on the cumulative heatmap data
		param: None
		return: a WindowGroup containing detected windows
		"""
		lim = self.__level*self.__factor
		return self.__map.make_windows(lim)


