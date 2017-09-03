import numpy as np
import tools_image as ti
import tools_perspective as tp
import tools_region as tr


class Region(object):
	"""
	Class to manage regions and perspective transformations
	"""
	def __init__(self, source, target, shape):
		assert source is not None and source.shape[0] == 4
		assert target is not None and target.shape[0] == 4
		assert shape is not None
		self.__source = source
		self.__target = target
		self.__shape = shape
		self.__transform = tp.Perspective(self.__source, self.__target) # create perspective transformation

	def get_source(self):
		return self.__source

	def get_target(self):
		return self.__target

	def get_transform(self):
		return self.__transform

	def draw(self, image, source, color=[255,0,0], linewidth=3):
		"""
		Draw the requested region onto the image
		param: image: the image into which the draw will be done
		param: source: boolean True for source region, False for target region
		"""
		dp = self.__shape if source == True else self.__target
		ti.draw_linepath(image, dp, color=color, linewidth=linewidth)

	def anchor(self):
		"""
		Extract and return an anchor 2-tuple from the target region
		"""
		return (self.__target[0][0], self.__target[-1][0])

	
class RegionBuilder(object):
	"""
	Class to manage the identification of appropriate source region
	"""
	def __init__(self, falpha=0.18, fbeta=0.60, fgamma=0.9,	ibeta=0.75, tbeta=0.8, tdelta=0.3, kernel_gauss=5, canny=(50, 200),
					drho=1, dtheta=np.pi/180.0, votes=50, min=20, gap=20):
		self.focus_alpha = falpha
		self.focus_beta = fbeta
		self.focus_gamma = fgamma
		self.inter_beta = ibeta
		self.target_beta = tbeta
		self.target_delta = tdelta
		self.kernel_gauss = kernel_gauss
		self.canny_limit = canny
		self.hough_rho = drho
		self.hough_theta = dtheta
		self.hough_votes = votes
		self.hough_min = min
		self.hough_gap = gap
		self.previous = None
		self.trace = None
		self.__scaler = ScaledImageSMinusH()

	def __extract_lines(self, frame_focus):
		n = 1 if self.previous is not None else 2
		for i in range(n):
			lines = ti.search_hough_lines(frame_focus, self.hough_rho, self.hough_theta, self.hough_votes//(i+1), self.hough_min, self.hough_gap)
			lines_lhs, lines_rhs =  tr.make_linesets(lines, frame_focus.shape)
			if len(lines_lhs) > 0 and len(lines_rhs) > 0: break
		return lines_lhs, lines_rhs

	def __blend_lines(self, lines_lhs, lines_rhs, shape):
		"""
		Make a set of two blended lines - order is important
		"""
		lines = tr.make_line_blend(lines_lhs, shape, self.inter_beta) + tr.make_line_blend(lines_rhs, shape, self.inter_beta)
		return lines

	def __sanity_check(self, region):
		if self.previous is not None:
			return tr.verify_region(region, self.previous.get_source())
		else:
			return True

	def __make_focus_main(self, frame):
		frame_gray = self.__scaler.make_scaled(frame)
		frame_blur = ti.make_gaussian_blur(frame_gray, kernelsize=self.kernel_gauss)
		frame_edge = ti.make_canny_transform(frame_blur, threshold=self.canny_limit)
		focus_poly = tr.make_focus_region(frame_edge.shape, self.focus_alpha, self.focus_beta, self.focus_gamma)
		frame_focus = ti.apply_region_mask(frame_edge, np.array([focus_poly]).astype(np.int32))
		return frame_focus

	def __make_focus_trace(self, frame):
		frame_gray = self.__scaler.make_scaled(frame)
		self.trace.append((frame_gray, "grayscale"))
		frame_blur = ti.make_gaussian_blur(frame_gray, kernelsize=self.kernel_gauss)
		self.trace.append((frame_blur, "gauss_blur"))
		frame_edge = ti.make_canny_transform(frame_blur, threshold=self.canny_limit)
		self.trace.append((frame_edge, "canny"))
		focus_poly = tr.make_focus_region(frame_edge.shape, self.focus_alpha, self.focus_beta, self.focus_gamma)
		frame_focus = ti.apply_region_mask(frame_edge, np.array([focus_poly]).astype(np.int32))
		self.trace.append((frame_focus, "focus"))
		return frame_focus

	def make_focus(self, frame):
		focus_poly = tr.make_focus_region(frame.shape, self.focus_alpha, self.focus_beta, self.focus_gamma)
		frame_focus = ti.apply_region_mask(frame, np.array([focus_poly]).astype(np.int32))
		return frame_focus

	def build_region(self, frame):
		"""
		Use the Canny-Hough line tools to identify lines, convert lines to a region for perspective transformation
		param: frame: the frame for which the region is to be computed (RGB)
		param: previous: the Region object from a previous frame (optional) returned if location procedure fails
		param: trace: optional list to which the sequence of image arrays will be appended
		return: Region object
		"""
		if self.trace is not None:
			frame_focus = self.__make_focus_trace(frame)
		else:
			frame_focus = self.__make_focus_main(frame)
		lines_lhs, lines_rhs = self.__extract_lines(frame_focus)
		final = self.__blend_lines(lines_lhs, lines_rhs, frame_focus.shape)
		if len(final) == 2:		# okay, we found enough lines
			reg_src = tr.make_region_from_lines(final)
			if self.__sanity_check(reg_src):
				reg_tgt = tr.make_target_region(frame_focus.shape, reg_src, self.target_beta, self.target_delta)
				return Region(reg_src, reg_tgt, frame_focus.shape)
			else:
				return self.previous
		else:
			if self.previous is not None:
				return self.previous			# assume the road hasn't changed too much
			else:
				if len(final) == 1:
					reg_src = tr.make_region_from_lines_flex(frame_focus.shape, final)		# use single line with reflection
				else:
					reg_src = focus_poly													# fallback to focus region
				reg_tgt = tr.make_target_region(frame_focus.shape, reg_src, self.target_beta, self.target_delta)
				return Region(reg_src, reg_tgt, frame_focus.shape)
	


class BinaryImage(object):
	"""
	Base class for producing search-ready binary images"
	"""
	def make_binary(self, image):
		raise NotImplementedError


class BinaryImageSobelSChannel(BinaryImage):
	"""
	Binary image using Sobel operator on S-channel
	"""
	def __init__(self, sobel_kernel=5, sobel_limit=(20,100), channel_limit=(160,255)):
		self.sobel_kernel = sobel_kernel
		self.sobel_limit = sobel_limit
		self.channel_limit = channel_limit

	def make_binary(self, image):
		channel_s = ti.select_space_hls(image)
		sobel_s_bin = ti.make_sobel_directed(channel_s, kernel=self.sobel_kernel, threshold=self.sobel_limit)
		channel_s_bin = ti.make_binary(channel_s, threshold=self.channel_limit)
		binary = np.zeros_like(sobel_s_bin)
		binary[(channel_s_bin == 1) | (sobel_s_bin == 1)] = 1.0
		return binary


class BinaryImageSobelSMHChannel(BinaryImage):
	"""
	Binary image using Sobel operator on S-channel minus H channel
	"""
	def __init__(self, sobel_kernel=5, sobel_limit=(20,100), channel_limit=(160,255)):
		self.sobel_kernel = sobel_kernel
		self.sobel_limit = sobel_limit
		self.channel_limit = channel_limit
		self.__scaler = ScaledImageSMinusH()

	def make_binary(self, image):
		channel_s = self.__scaler.make_scaled(image)
		sobel_s_bin = ti.make_sobel_directed(channel_s, kernel=self.sobel_kernel, threshold=self.sobel_limit)
		channel_s_bin = ti.make_binary(channel_s, threshold=self.channel_limit)
		binary = np.zeros_like(sobel_s_bin)
		binary[(channel_s_bin == 1) | (sobel_s_bin == 1)] = 1.0
		return binary


class ScaledImage(object):
	"""
	Base class for producing scaled images
	"""
	def make_scaled(self, image):
		raise NotImplementedError


class ScaledImageSMinusH(ScaledImage):
	"""
	Scaled image as HLS S-channel minus HLS H channel
	"""
	def __init__(self, channels="H,S"):
		self.__channels = channels

	def make_scaled(self, image):
		chan_h, chan_s = ti.select_space_hls(image, space=self.__channels)
		return np.where(chan_s > chan_h, chan_s - chan_h, 0)