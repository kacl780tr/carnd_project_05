import pickle
import numpy as np
import skimage.io as skio
import skimage.exposure as skex
import skimage.util as skul
import sklearn
import sklearn.cross_validation
import os
from pathlib import Path
import pandas as pd
import display
import tools_window as twdw
import warnings


def rescale_channel(batch, convert_to=np.float32):
	"""
	Rescale all channels of an image to (-1, 1) independently
	param: batch: a 4D array of immages with shape [index, height, width, depth]
	return: rescaled images with data type np.float32
	"""
	batch = batch.astype(np.float32)
	min = np.min(batch, axis=(1,2), keepdims=True)		# not sure this is a good idea, might cause color balance issues
	max = np.max(batch, axis=(1,2), keepdims=True)
	mu = 0.5*(max + min)
	rg = 0.5*(max - min + 1.0e-6)
	if convert_to == np.float32: return batch
	return batch.astype(convert_to)


def rescale_flat(batch, convert_to=np.float32):
	"""
	Rescale all channels of an image to (-1, 1) simultaneously
	param: batch: a 4D array of immages with shape [index, height, width, depth]
	return: rescaled images with data type np.float32
	"""
	batch = batch.astype(np.float32)
	min = np.min(batch, axis=(1,2,3), keepdims=True)
	max = np.max(batch, axis=(1,2,3), keepdims=True)
	mu = 0.5*(max + min)
	rg = 0.5*(max - min + 1.0e-6)
	batch = (batch - mu)/rg
	if convert_to == np.float32: return batch
	return batch.astype(convert_to)


def rescale_pass(batch, convert_to=np.float32):
	if convert_to is not np.float32:
		return batch.astype(convert_to)
	else:
		return batch


class DataSet(object):
	"""
	A class for self-contained handling of datasets
	"""
	def __init__(self, feature, label, scaler=None, output_dtype=(np.float32, np.int32), preserve_order=False):
		"""
		Initialize
		param: feature: data features here assumed to be a numpy 2d array
		param: labels: data labels here assumed to be a numpy 1d array
		"""
		assert len(feature) == len(label), "feature-label mismatch"
		self.feature = feature
		self.label = label
		self.length = len(self.feature)
		self.__preserve = preserve_order
		self.__data = None
		self.__scaler = scaler
		if self.__scaler is None:
			self.__scaler = rescale_pass
		self.feature_dtype = output_dtype[0]
		self.label_dtype = output_dtype[1]

	def feature_shape(self):
		return self.feature[0].shape

	def shuffle(self):
		if self.__preserve and self.__data is None:
			self.__data = (self.feature, self.label)
		self.feature, self.label = sklearn.utils.shuffle(self.feature, self.label)
		return self

	def reset(self):
		if self.__data is not None:
			self.feature = self.__data[0]
			self.label = self.__data[1]
			self.__data = None
		return self

	def batch_count(self, size_batch):
		assert size_batch >= 0, "invalid batch size"
		if size_batch == 0: return 1
		n = self.length//size_batch
		if self.length % size_batch:
			n += 1
		return n

	def batch_make(self, size_batch):
		if size_batch == 0:
			return self.__scaler(self.feature, convert_to=self.feature_dtype), self.label.astype(self.label_dtype), self.length
		for b in range(0, self.length, size_batch):
			x = self.feature[b:b+size_batch]
			y = self.label[b:b+size_batch]
			yield self.__scaler(x, convert_to=self.feature_dtype), y.astype(self.label_dtype), len(x)

	def batch_select(self, size_batch, index_begin=0):
		x = self.feature[index_begin:index_begin+size_batch]
		y = self.label[index_begin:index_begin+size_batch]
		return self.__scaler(x, convert_to=self.feature_dtype), y.astype(self.label_dtype), len(x)


class DataFrame(object):
	"""
	A class representing prediction data or a dummy DataSet
	"""
	def __init__(self, feature=None, shape=None, scaler=None, output_dtype=(np.float32, np.int32)):
		assert shape is not None or feature is not None, "must provide an example or shape"
		self.feature = feature
		self.__set_shape(shape)
		if self.feature is not None:
			self.__set_shape(self.feature.shape)
		self.label = None
		self.length = 1
		self.__scaler = scaler
		if self.__scaler is None:
			self.__scaler = rescale_pass
		self.feature_dtype = output_dtype[0]
		self.label_dtype = output_dtype[1]

	def __set_shape(self, shape):
		if shape is not None:
			self.__shape = shape
		if len(self.__shape) > 3:
			self.__shape = self.__shape[1:]

	def feature_shape(self):
		return self.__shape

	def shuffle(self):
		return self

	def reset(self):
		return self

	def batch_count(self, size_batch):
		assert size_batch >= 0, "invalid batch size"
		return 0

	def batch_make(self, size_batch, convert_to=np.float32):
		raise NotImplementedError

	def batch_select(self, size_batch, index_begin=0, convert_to=np.float32):
		raise NotImplementedError


def read_images(pathlist):
	if len(pathlist) == 0: return None
	images = skio.imread(pathlist[0])
	images = np.zeros((len(pathlist), *images.shape), dtype=images.dtype)
	for i, pth in enumerate(pathlist):
		img = skio.imread(pth)
		images[i,:,:,:] = img
	return images


def compile_pathlist(dir_source, shuffle=True):
	subdirs = os.listdir(dir_source)
	if shuffle:
		def selector(subdir):
			fn = os.listdir(subdir)
			np.random.shuffle(fn)
			for i in range(len(fn)):
				path = os.path.join(subdir, fn[i])
				yield path
		subdir_gen = []
		for sd in subdirs:
			sd_path = os.path.join(dir_source, sd)
			subdir_gen.append(selector(sd_path))
		path_list = []
		while len(subdir_gen) > 0:
			for i in range(len(subdir_gen)):
				try:
					sub_path = next(subdir_gen[i])
				except StopIteration:
					subdir_gen.remove(subdir_gen[i])
					break;
				path_list.append(sub_path)
		return path_list
	else:
		path_list = []
		for sd in subdirs:
			sd_path = os.path.join(dir_source, sd)
			sd_list = os.listdir(sd_path)
			for fn in sd_list:
				path = os.path.join(sd_path, fn)
				path_list.append(path)
		return path_list


def list_directory(dir_source):
	files = os.listdir(dir_source)
	pathlist = []
	for fn in files:
		pathlist.append(os.path.join(dir_source, fn))
	return pathlist


def archive_dump(filename, dict):
	path = Path(filename)
	if path.exists():
		path.unlink()
	with open(filename, mode="wb") as tgt:
		pickle.dump(dict, tgt)


def archive_load(filename):
	path = Path(filename)
	if path.exists():
		with open(filename, mode="rb") as src:
			archive = pickle.load(src)
			return archive
	else:
		return None


def augment_data(source):
	images = source["images"]
	labels = source["labels"]
	flipped = np.zeros_like(images)
	for i in range(images.shape[0]):
		img = images[i,:,:,:]
		img = np.fliplr(img)
		flipped[i,:,:,:] = img
	target = {
		"images": flipped,
		"labels": labels
		}
	return target


def concatenate_data(datalist):
	if len(datalist) == 0: return None
	if len(datalist) == 1: return datalist[0]
	data_full = datalist[0]
	for dat in datalist[1:]:
		images = np.concatenate([data_full["images"], dat["images"]])
		labels = np.concatenate([data_full["labels"], dat["labels"]])
		data_full["images"] = images
		data_full["labels"] = labels
	return data_full


def load_base_data(filenames):
	datasets = []
	for fn in filenames:
		data = archive_load(fn)
		datasets.append(data)
	return concatenate_data(datasets)


def make_dataset(filename, extractor, frac_test=0.2, frac_valid=0.0, scaler=None):
	assert extractor is not None, "invalid feature extractor"
	assert frac_test >= 0 and frac_test <= 1.0, "invalid test fraction"
	assert frac_valid >= 0 and frac_valid <= 1.0, "invalid validation fraction"
	image_data = load_base_data(filename)
	features = extractor.extract_features(image_data["images"])
	labels = image_data["labels"]
	if frac_test > 0.0:
		features_train, features_test, labels_train, labels_test = sklearn.cross_validation.train_test_split(features, labels, test_size=frac_test, stratify=labels, random_state=4357)
		features_train = extractor.process_features(features_train)
		features_test = extractor.process_features(features_test)
		if frac_valid > 0.0:
			features_train, features_valid, labels_train, labels_valid = sklearn.cross_validation.train_test_split(features_train, labels_train, test_size=frac_valid, stratify=labels_train)
			return DataSet(features_train, labels_train, scaler=scaler), DataSet(features_valid, labels_valid, scaler=scaler), DataSet(features_test, labels_test, scaler=scaler)
		return DataSet(features_train, labels_train, scaler=scaler), DataSet(features_test, labels_test, scaler=scaler)
	else:
		features = extractor.process_features(features)
		return DataSet(features, labels, scaler=scaler)


def compile_basic_data():
	path_vehicles = "./input/vehicles"
	path_nonvehicles = "./input/non-vehicles"
	path_list = compile_pathlist(path_vehicles)
	path_list += compile_pathlist(path_nonvehicles)
	np.random.shuffle(path_list)
	labels = np.zeros_like(path_list, dtype=np.uint8)
	for i, pth in enumerate(path_list):
		if pth.startswith(path_vehicles):
			labels[i] = 1
	images = read_images(path_list)
	archive = {
		"images": images,
		"labels": labels
		}
	archive_dump("./detection.p", archive)


def compile_extra_data(size, show=False):
	path_source = "./object-detection-crowdai/"
	label_file = "labels.csv"
	label_path = os.path.join(path_source, label_file)
	label_frame = pd.read_csv(label_path, header=0)
	print("Full data set contains {} rows".format(len(label_frame)))
	label_data = label_frame[(label_frame["Label"] != "Pedestrian")].reset_index()
	label_data = label_data.drop("index", 1).drop("Preview URL", 1)
	label_data.columns = ["xmin", "ymin", "xmax", "ymax", "Frame", "Label"]		# we re-label columns to correct values
	n = len(label_data)
	img_list = list(label_data["Frame"].unique())
	n_img = len(img_list)
	print("There are {} labeled boxes in the dataset".format(n))
	print("There are {} unique images in the dataset".format(n_img))
	print(label_data.head(10))
	win_build = twdw.WindowBuilder(ceiling=1)
	images_car, images_non = [], []
	def select_image():
		k = np.random.randint(0, len(img_list))
		nm = img_list[k]
		img_list.remove(nm)
		img_frame = label_data[label_data.Frame == nm]
		return nm, img_frame
	def make_window(shape):
		xlo = np.random.randint(shape[1]//8, 7*shape[1]//8)
		xhi = xlo + 128
		ylo = np.random.randint(300, 750)
		yhi = ylo + 128
		return twdw.Window(xlo, ylo, xhi, yhi)
	def sample_non_car(image, win_all):							# sample non-car boxes
		for i in range(10):
			if len(images_non) == len(images_car):				# ok, we have same number as cars
				break
			rw = make_window(image.shape)						# make random window
			image_tmp = np.copy(image_src)
			if show:
				rw.draw(image_tmp, color=(255, 0, 0), width=2)
				display.display_image(image_tmp)
			if win_all.has_overlap(rw, image.shape): continue								# check for overlap
			img_non = rw.extract_scale(image)
			if show: display.display_image(img_non)
			images_non.append(img_non)
	while len(images_car) < size or len(images_non) < size:
		if len(img_list) == 0: break;
		image_name, image_frame = select_image()											# select image at random
		image_path = os.path.join(path_source, image_name)
		image_src = skio.imread(image_path)
		win_car, win_all = twdw.WindowGroup(), twdw.WindowGroup()
		for i, row in image_frame.iterrows():												# compile WindowGroup containing all boxes for this image
			label = row["Label"]
			xmin, xmax, ymin, ymax = row["xmin"], row["xmax"], row["ymin"], row["ymax"]
			try:
				wdw = twdw.Window(min(xmin, xmax), min(ymin,ymax), max(xmin, xmax), max(ymin, ymax))
				win_all.append(wdw)
				if label == "Car": win_car.append(wdw)
			except AssertionError:
				pass
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			image_eqn = skex.equalize_adapthist(image_src)
			image_eqn = skul.img_as_ubyte(image_eqn)
		win_car.sort(reverse=True)															# sort largest -> smallest
		ws = (0, 0)
		if len(win_car) > 0 and len(images_car) < size:
			if show:
				win_all.draw(image_src, width=2)
				win_car[0].draw(image_src, color=(255,191,0), width=2)
				display.display_image(image_src)
			ws = win_car[0].get_span()
		if ws[0] < 128 or ws[1] < 128:														# make sure biggest window is large enough
			sample_non_car(image_eqn, win_all)
		else:
			images_car.append(win_car[0].extract_scale(image_eqn))
			img_sub = win_car[0].extract_scale(image_eqn, size=(128, 128))
			if show: display.display_image(img_sub)
			windows = win_build.make_window_group(img_sub.shape)
			for wdw in windows:
				img_w = wdw.extract_scale(img_sub)
				images_car.append(img_w)
				if show: display.display_image(img_w)
			if len(images_car) - len(images_non) > 5:							# try to step up non-car samples
				sample_non_car(image_eqn, win_all)
		print("Processed image {}, car images = {}/{}, non-car = {}/{}".format(image_path, len(images_car), size, len(images_non), size))
	feature_car, feature_non = None, None
	label_car, label_non = None, None
	if len(images_car) > 0:
		feature_car = np.zeros((len(images_car), *images_car[0].shape), dtype=images_car[0].dtype)
		for i, img in enumerate(images_car):
			feature_car[i,:,:,:] = img
		label_car = np.ones((feature_car.shape[0]), dtype=feature_car.dtype)
	if len(images_non) > 0:
		feature_non = np.zeros((len(images_non), *images_non[0].shape), dtype=images_non[0].dtype)
		for i, img in enumerate(images_non):
			feature_non[i,:,:,:] = img
		label_non = np.zeros((feature_non.shape[0]), dtype=feature_non.dtype)
	data_dict = {}
	if feature_car is not None:
		data_dict["images"] = feature_car
		data_dict["labels"] = label_car
	if feature_non is not None:
		if "images" in data_dict.keys():
			data_dict["images"] = np.concatenate((data_dict["images"], feature_non))
		else:
			data_dict["images"] = feature_non
		if "labels" in data_dict.keys():
			data_dict["labels"] = np.concatenate((data_dict["labels"], label_non))
		else:
			data_dict["labels"] = label_non
	return data_dict


def sample_images(images, labels):
	"""
	Select a sample image from images for each unique label in labels
	param: images: the source image set
	labels: the labels corresponding to the images
	"""
	label_set = list(set(labels))
	selection = []
	for lbl in label_set:
		available = [i for i,lb in enumerate(labels) if lb == lbl]
		k = np.random.choice(available)
		selection.append(images[k])
	return selection


def data_summary(train, valid=None, test=None):
	print("Number of training examples = {}".format(train.length))
	if valid is not None: print("Number of validation examples = {}".format(valid.length))
	if test is not None: print("Number of testing examples = {}".format(test.length))
	print("Feature shape = {}".format(train.feature_shape()))


if __name__ == "__main__":
	#dataset = compile_extra_data(5000)
	#archive_dump("./detection_extra.p", dataset)
	data = archive_load("./detection.p")
	samples = sample_images(data["images"], data["labels"])
	samples.extend(sample_images(data["images"], data["labels"]))
	display.display_image_grid(samples, title="Sample images", subtitle=["Non-vehicle", "Vehicle", "Non-vehicle", "Vehicle"], ncol=2)
