import numpy as np
import calibration as cam
import stage_lane as slne
import stage_detect as sdet
import tools_image as tim


class Configuration(object):
	def __init__(self):
		self.trace = False
		self.trace_handler = None
		self.camera_file = "./calibration.p"
		self.enable_lane = True
		self.svm_file = "./model_svm/svm_yuv_cs2_06_sp06_hog0a5-16+2.p"										# for loading trained svm model and feature extractor
		self.window_rules = rules=[(1.0, 0.55, 0.65), (1.5, 0.60,0.8), (2.0,0.60,0.92)]						# rules to generate search windows (scale, alpha, beta)
		self.window_overlap = (0.5, 0.5)																	# search window overlap
		self.window_width = 2																				# line width for drawing detection windows
		self.map_length = 10
		self.map_factor = 0.6
		self.cnn_file = "./model_cnn/model.ckpt"															# for loading trained cnn model
		self.rate_leak = 0.08786																			# for instantiating prediction graph
		self.threshold = 0.989																				# detection probability threshold
		self.frame_shape = (720, 1280, 3)																	# frame size



class Pipeline(object):
	"""
	Class to manage the processing pipeline
	"""
	def __init__(self, configuration):
		assert configuration is not None, "Pipeline configuration not available" 
		self.__config = configuration
		self.__camera = cam.calibration_retrieve(self.__config.camera_file)
		assert self.__camera is not None, "Camera calibration unavailable in file {}".format(self.__config.camera_file)
		self.__lane = slne.LaneBuilder(enable=self.__config.enable_lane)
		#self.__detect = sdet.DetectionBuilderSVM(self.__config.svm_file, rules=self.__config.window_rules, overlap=self.__config.window_overlap, map_count=self.__config.map_length, map_factor=self.__config.map_factor)
		self.__detect = sdet.DetectionBuilderCNN(self.__config.cnn_file, rate_leak=self.__config.rate_leak, threshold=self.__config.threshold, shape=self.__config.frame_shape, map_count=self.__config.map_length, map_factor=self.__config.map_factor)
		self.__trace = None
		if self.__config.trace and self.__config.trace_handler is not None:
			self.__config.trace_handler.setup()
			self.__trace = []												# set an empty list to retain processing steps
			self.__lane.trace = self.__trace								# also set for lane builder
			self.__detect.trace = self.__trace								# and for detection builder

	def __trace_reset(self):
		if self.__trace is not None:
			self.__trace = []
			self.__lane.trace = self.__trace
			self.__detect.trace = self.__trace

	def __process_main(self, frame):
		frame_corr = self.__camera.apply_correction(frame)									# apply camera correction
		frame_lane = self.__lane.process(frame_corr)										# perform lane detection
		detection = self.__detect.build_detection(frame_corr)								# perform vehicle detection
		detection.draw(frame_lane, width=self.__config.window_width)
		return frame_lane

	def __process_trace(self, frame):
		self.__trace.append((frame, "raw"))
		frame_corr = self.__camera.apply_correction(frame)									# apply camera correction
		self.__trace.append((frame_corr, "corrected"))
		frame_lane = self.__lane.process(frame_corr)
		final = np.copy(frame_lane)
		detection = self.__detect.build_detection(frame_corr)
		detection.draw(final, width=self.__config.window_width)
		self.__trace.append((final, "final"))
		if self.__config.trace_handler is not None:
			self.__config.trace_handler.run(self.__trace)
		return final

	def has_trace(self):
		return bool(self.__trace is not None)

	def get_trace(self):
		return self.__trace

	def process(self, frame):
		if self.has_trace():
			self.__trace_reset()																# reset the trace
			return self.__process_trace(frame)
		else:
			return self.__process_main(frame)

	def reset_pipeline(self):
		self.__lane.reset()
		self.__detect.reset()
		return self



class TraceHandler(object):
	def setup(self):
		raise NotImplementedError

	def run(self, trace):
		raise NotImplementedError
		