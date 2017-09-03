import numpy as np
import tools_window as tw
import tools_svm as tsvm
import tools_net as tnet
import tools_tflow as tflw
import tools_data as tdat


class DetectionBuilder(object):
	"""
	Base class to manage the process of object detection from an image
	"""
	def __init__(self):
		super().__init__()
		self.trace = None

	def build_detection(self, frame):
		raise NotImplementedError

	def reset(self):
		raise NotImplementedError
		return self


class DetectionBuilderSVM(DetectionBuilder):
	def __init__(self, modelfile, baseline=(64,64), rules=[(1.0, 0.0, 1.0)], overlap=(0.5, 0.5), x_stretch=1.2, ceiling=0, map_count=10, map_factor=0.5, shape=None):
		super().__init__()
		self.__bldr = tw.WindowBuilder(baseline=baseline, rules=rules, overlap=overlap, x_stretch=x_stretch, ceiling=ceiling)
		self.__clsr = None
		self.__extr = None
		assert modelfile is not None, "invalid model file"
		try:
			self.__clsr, self.__extr = tsvm.model_load(modelfile)
		except Exception:
			pass
		assert self.__clsr is not None and self.__extr is not None, "unable to read modelfile {}".format(modelfile)
		self.__map_count = map_count
		self.__map_factor = map_factor
		self.__windows = None
		self.__heatmap = None
		self.__last = None

	def __initialize(self, shape):
		self.__windows = self.__bldr.make_window_group(shape)
		self.__heatmap = tw.HeatMapGroup(shape, count=self.__map_count, limit_factor=self.__map_factor)
		if self.trace is not None:
			print("Using {} search windows".format(len(self.__windows)))

	def __get_heatmap(self):
		if self.__last is None:
			return tw.HeatMap(self.__heatmap.get_shape())
		else:
			return self.__last.reset()

	def __make_detection_trace(self, frame):
		frame_data = self.__windows.extract_scale(frame)
		frame_search = np.copy(frame)
		self.__windows.draw(frame_search, color=(255, 0, 0))						# red
		self.trace.append((frame_search, "window_search"))
		frame_feat = self.__extr.extract_features(frame_data)
		frame_feat = self.__extr.process_features(frame_feat)
		predict = self.__clsr.predict(frame_feat)
		n_detect = np.sum(predict)
		print("Detection count = {}".format(n_detect))
		frame_predict = np.copy(frame)
		self.__windows.draw(frame_predict, indicator=predict, color=(255, 192,0))	# amber
		self.trace.append((frame_predict, "window_predict"))
		map = self.__get_heatmap()
		self.__windows.update_heatmap(map.get_map(), indicator=predict)
		self.trace.append((np.copy(map.get_map()), "window_heatmap"))
		self.__last = self.__heatmap.add_map(map)
		self.trace.append((np.copy(self.__heatmap.get_map().get_map()), "cumulative_heatmap"))
		window_detect = self.__heatmap.make_windows()
		frame_detect = np.copy(frame)
		window_detect.draw(frame_detect, color=(0, 255, 0))							# green
		self.trace.append((frame_detect, "window_track"))
		return window_detect

	def __make_detection(self, frame):
		frame_data = self.__windows.extract_scale(frame)
		frame_feat = self.__extr.extract_features(frame_data)
		frame_feat = self.__extr.process_features(frame_feat)
		predict = self.__clsr.predict(frame_feat)
		map = self.__get_heatmap()
		self.__windows.update_heatmap(map.get_map(), indicator=predict)
		self.__last = self.__heatmap.add_map(map)
		return self.__heatmap.make_windows()

	def build_detection(self, frame):
		if self.__heatmap is None: self.__initialize(frame.shape)
		if self.trace is not None:
			return self.__make_detection_trace(frame)
		else:
			return self.__make_detection(frame)

	def reset(self):
		if self.__heatmap is not None:
			self.__heatmap.reset()
		self.__last = None
		return self


class DetectionBuilderCNN(DetectionBuilder):
	def __init__(self, modeldir, rate_leak=None, threshold=0.5, shape=(1280, 720), baseline=(64,64), limit_v=(0.55, 0.92), map_count=10, map_factor=0.5):
		super().__init__()
		assert modeldir is not None, "invalid model directory"
		assert rate_leak is not None, "invalid leak rate"
		self.__model = modeldir																# directory containing model checkpoint						
		self.__base = baseline
		self.__limit = limit_v
		self.__leak = rate_leak
		self.__floor = threshold
		self.__shape = shape																# input image shape
		self.__slice = None																	# slice of image submitted to classifier
		self.__windows = None																# window group
		self.__clsr = None																	# neural net model runner
		self.__make_windows()																# setup the map
		self.__make_runner()																# setup the model runner
		self.__heatmap = tw.HeatMapGroup(self.__shape, map_count, limit_factor=map_factor)	# heatmap group across frames
		self.__last = None

	def __make_windows(self):
		self.__windows = tw.WindowGroup()
		sy = int(self.__shape[0]*self.__limit[0])
		dy = int(self.__shape[0]*self.__limit[1]) - sy - self.__base[1]
		for j in range(27):
			ul_y = j*8																		# 8 comes from stride of conv_4
			for i in range(153):
				ul_x = i*8
				wdw = tw.Window(ul_x, ul_y + sy, ul_x + self.__base[0], ul_y + sy + self.__base[1])
				self.__windows.append(wdw)
		self.__slice = (sy, sy + dy + self.__base[1])

	def __make_runner(self):
		def make_predict():
			dy = self.__slice[1] - self.__slice[0]
			data = tdat.DataFrame(shape=(dy, self.__shape[1], 3))
			return tflw.make_predictor(data, self.__leak)
		self.__clsr = tnet.ModelRunner(model_path=self.__model, graph_call=make_predict)

	def __get_heatmap(self):
		if self.__last is None:
			return tw.HeatMap(self.__heatmap.get_shape())
		else:
			return self.__last.reset()

	def __make_detection_trace(self, frame):
		frame_search = np.copy(frame)
		self.__windows.draw(frame_search, color=(255, 0, 0))									# red
		self.trace.append((frame_search, "window_search"))
		frame_slice = frame[self.__slice[0]:self.__slice[1],:,:]								# select active image slice
		frame_slice = np.expand_dims(frame_slice, axis=0)										# add batch dimension
		frame_slice = tdat.rescale_channel(frame_slice)											# normalize data
		predict = self.__clsr.predict(frame_slice)												# get predictions
		predict[predict < self.__floor] = 0.0
		frame_predict = np.copy(frame)
		self.__windows.draw(frame_predict, indicator=predict.ravel(), color=(255,192,0))	# amber
		self.trace.append((frame_predict, "window_predict"))
		map = self.__get_heatmap()
		self.__windows.update_heatmap(map.get_map(), indicator=predict.ravel())
		self.trace.append((np.copy(map.get_map()), "window_heatmap"))
		self.__last = self.__heatmap.add_map(map)
		self.trace.append((np.copy(self.__heatmap.get_map().get_map()), "cumulative_heatmap"))
		window_detect = self.__heatmap.make_windows()
		frame_detect = np.copy(frame)
		window_detect.draw(frame_detect, color=(0,255,0))										# green
		self.trace.append((frame_detect, "window_track"))
		return window_detect

	def __make_detection(self, frame):
		frame_slice = frame[self.__slice[0]:self.__slice[1],:,:]								# select active image slice
		frame_slice = np.expand_dims(frame_slice, axis=0)										# add batch dimension
		frame_slice = tdat.rescale_channel(frame_slice)											# normalize data
		predict = self.__clsr.predict(frame_slice)												# get predictions
		predict[predict < self.__floor] = 0.0													# select to threshold
		map = self.__get_heatmap()
		self.__windows.update_heatmap(map.get_map(), indicator=predict.ravel())
		self.__last = self.__heatmap.add_map(map)
		return self.__heatmap.make_windows()

	def build_detection(self, frame):
		if self.trace is not None:
			return self.__make_detection_trace(frame)
		else:
			return self.__make_detection(frame)

	def reset(self):
		if self.__heatmap is not None:
			self.__heatmap.reset()
		self.__last = None
		return self