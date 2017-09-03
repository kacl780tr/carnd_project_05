import numpy as np
import stage_region as sreg
import stage_path as spth
import tools_image as tim


class LaneBuilder(object):
	def __init__(self, enable=True):
		self.__region = sreg.RegionBuilder()
		self.__path = spth.PathFunctionBuilder()
		self.__binary = sreg.BinaryImageSobelSMHChannel()		# setup the binary image maker
		self.__enable = enable
		self.trace = None

	def __add_info(self, frame, path_function):
		dev = path_function.deviation()
		crv = path_function.realspace_path().curvature()
		text = "lane curvature {:>8.0f}m - deviation {:>2.2f}m to the {}"
		if dev[1] >= 0:
			side = "right"
		else:
		   side = "left"
		tim.draw_textpath(frame, text.format(crv, np.abs(dev[1]), side))

	def __process_main(self, frame):
		region = self.__region.build_region(frame)
		self.__region.previous = region														# set region for use with next frame
		binary = self.__binary.make_binary(frame)
		binary_focus = self.__region.make_focus(binary)
		binary_tfm = region.get_transform().apply(binary_focus)
		path_function = self.__path.build_path(binary_tfm, anchor_points=region.anchor())	# extract path function from binary image
		self.__path.previous_path = path_function											# set region for use with next frame
		path_tfm = path_function.draw()														# draw path function template
		path_norm = region.get_transform().unapply(path_tfm)
		final = tim.make_overlay(frame, path_norm)
		self.__add_info(final, path_function)
		return final

	def __process_trace(self, frame):
		region = self.__region.build_region(frame)
		self.__region.previous = region														# set region for use with next frame
		frame_region = np.copy(frame)
		tim.draw_linepath(frame_region, region.get_source())
		self.trace.append((frame_region, "region"))
		binary = self.__binary.make_binary(frame)
		self.trace.append((binary, "binary"))
		binary_focus = self.__region.make_focus(binary)									
		self.trace.append((binary_focus, "binary_focus"))
		binary_tfm = region.get_transform().apply(binary_focus)
		self.trace.append((binary_tfm, "binary_transform"))
		path_function = self.__path.build_path(binary_tfm, anchor_points=region.anchor())	# extract path function from binary image
		self.__path.previous_path = path_function											# set region for use with next frame
		path_tfm = path_function.draw()
		self.trace.append((path_tfm, "path_transform"))
		path_overlay = region.get_transform().unapply(path_tfm)
		self.trace.append((path_overlay, "path_normal"))
		final = tim.make_overlay(frame, path_overlay)
		self.__add_info(final, path_function)
		self.trace.append((final, "lane_final"))
		return final

	def reset(self):
		self.__region.previous = None
		self.__path.previous_path = None
		return self

	def process(self, frame):
		if not self.__enable: return frame
		if self.trace is not None:
			return self.__process_trace(frame)
		else:
			return self.__process_main(frame)

