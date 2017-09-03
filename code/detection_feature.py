import numpy as np
import cv2
import skimage.feature as skfr
import sklearn.preprocessing
import sklearn.decomposition


class FeatureExtractor(object):
	"""
	Base class for feature extraction
	"""
	def __init__(self):
		self.reset()

	def extract_features(self, imageset):
		raise NotImplementedError

	def reset(self):
		self.__scaler = None

	def process_features(self, features):
		if self.__scaler is None:
			self.__scaler = sklearn.preprocessing.StandardScaler()
			self.__scaler.fit(features)
		return self.__scaler.transform(features)


class FeatureExtractorPassthru(FeatureExtractor):
	def __init__(self):
		super().__init__()

	def extract_features(self, imageset):
		return imageset

	def process_features(self, features):
		return features


class FeatureExtractorBasic(FeatureExtractor):
	def __init__(self, use_spatial=True, use_color=True, use_hog=True, spatial_size=(32, 32), color_space="RGB", color_channel=-1, bin_count=32, bin_range=(0, 255), hog_channel=-1, n_angle=9, pix_per_cell=8, cell_per_block=2):
		super().__init__()
		self.spatial_active = use_spatial
		self.__size = spatial_size
		self.color_active = use_color
		self.__convert = self.__get_convert(color_space)
		self.__channel = color_channel
		self.__bin_n = bin_count
		self.__bin_r = bin_range
		self.hog_active = use_hog
		self.__hog_chan = hog_channel
		self.__hog_ang = n_angle
		self.__hog_ppc = (pix_per_cell, pix_per_cell)
		self.__hog_cpb = (cell_per_block, cell_per_block)
		self.__dtype = np.float32
		assert self.spatial_active or self.color_active or self.hog_active, "invalid configuration - no active features"

	def __get_convert(self, space):
		cs = space.strip().lower()
		convert = None
		if cs == "hls":
			convert = cv2.COLOR_RGB2HLS
		elif cs == "hsv":
			convert = cv2.COLOR_RGB2HSV
		elif cs == "luv":
			convert = cv2.COLOR_RGB2LUV
		elif cs == "yuv":
			convert = cv2.COLOR_RGB2YUV
		elif cs == "ycrcb":
			convert = cv2.COLOR_RGB2YCrCb
		return convert

	def __convert_color(self, image_rgb):
		if self.__convert is not None:
			return cv2.cvtColor(image_rgb, self.__convert)
		else:
			return image_rgb

	def __make_spatial(self, image):
		feat_spatial = cv2.resize(image, self.__size, interpolation=cv2.INTER_NEAREST)
		return feat_spatial.ravel()

	def __make_histogram(self, image):
		if len(image.shape) < 3 or image.shape[2] == 1:
			return np.histogram(image, bins=self.__bin_n, range=self.__bin_r)[0]
		elif self.__channel != -1:
			return np.histogram(image[:,:,self.__channel], bins=self.__bin_n, range=self.__bin_r)[0]
		else:
			hist_sections = []
			for c in range(image.shape[2]):
				hist_channel = np.histogram(image[:,:,c], bins=self.__bin_n, range=self.__bin_r)
				hist_sections.append(hist_channel[0])
			return np.concatenate(hist_sections)

	def __make_hog(self, image):
		if len(image.shape) < 3 or image.shape[2] == 1:
			return skfr.hog(image, self.__hog_ang, self.__hog_ppc, self.__hog_cpb, visualise=False, feature_vector=True)
		elif self.__hog_chan != -1:
			return skfr.hog(image[:,:, self.__hog_chan], self.__hog_ang, self.__hog_ppc, self.__hog_cpb, visualise=False, feature_vector=True)
		else:
			hog_sections = []
			for c in range(image.shape[2]):
				hog_sect = skfr.hog(image[:,:,c], self.__hog_ang, self.__hog_ppc, self.__hog_cpb, visualise=False, feature_vector=True)
				hog_sections.append(hog_sect)
			return np.concatenate(hog_sections)

	def __extract_feature(self, image):
		image_cs = self.__convert_color(image)
		sections = []
		if self.spatial_active:
			feat_spatial = self.__make_spatial(image_cs)
			sections.append(feat_spatial.astype(self.__dtype))
		if self.color_active:
			feat_color = self.__make_histogram(image_cs)
			sections.append(feat_color.astype(self.__dtype))
		if self.hog_active:
			feat_hog = self.__make_hog(image_cs)
			sections.append(feat_hog.astype(self.__dtype))
		if len(sections) == 1:
			return sections[0]
		else:
			return np.concatenate(sections)
	
	def __call__(self, image):
		return self.__extract_feature(image)

	def extract_features(self, imageset):
		if imageset is None: return None
		shape = imageset.shape
		if len(shape) == 3:							# single image
			return self.__extract_feature(imageset)
		img = imageset[0,:,:,:]
		tst = self.__extract_feature(img)
		features = np.zeros((shape[0], *tst.shape), dtype=self.__dtype)
		for i in range(shape[0]):
			img = imageset[i,:,:,:]
			features[i,:] = self.__extract_feature(img)
		return features


class FeatureExtractorPCA(FeatureExtractorBasic):
	def __init__(self, use_spatial=True, use_color=True, use_hog=True, pca_reduction=10, spatial_size=(32, 32), color_space="RGB", color_channel=-1, bin_count=32, bin_range=(0, 255), hog_channel=-1, n_angle=9, pix_per_cell=8, cell_per_block=2):
		super().__init__(use_spatial, use_color, use_hog, spatial_size, color_space, color_channel, bin_count, bin_range, hog_channel, n_angle, pix_per_cell, cell_per_block)
		self.__pca_factor = pca_reduction
		self.__pca_decomp = None

	def process_features(self, features):
		if self.__pca_decomp is None:
			dim = int(max(1, features.shape[1]//self.__pca_factor))
			self.__pca_decomp = sklearn.decomposition.PCA(n_components=dim)
			self.__pca_decomp.fit(features)
			evr = np.sum(self.__pca_decomp.explained_variance_ratio_)
			print("FeatureExtractorPCA: result dimension = {}, explained variance ratio = {:>1.6f}".format(dim, evr))
		return super().process_features(self.__pca_decomp.transform(features))


class FeatureExtractorORB(FeatureExtractor):
	def __init__(self, count=16):
		super().__init__()
		self.__count = count
		self.__orb = cv2.ORB_create(nfeatures=self.__count, edgeThreshold=5, nlevels=10, fastThreshold=5)
		self.__dtype = np.float32
		self.__incr = 0

	def __extract_feature(self, image):
		keypts = self.__orb.detect(image)
		keypts, des = self.__orb.compute(image, keypts)
		if des is None:
			self.__incr += 0.01
			return np.ones((self.__count*32))*self.__incr
		if len(keypts) < self.__count:								# require uniform size for downstream
			n = self.__count - len(keypts)
			des = np.vstack((des, np.zeros((n, des.shape[-1]))))
		return des.ravel()

	def extract_features(self, imageset):
		if imageset is None: return None
		shape = imageset.shape
		if len(shape) == 3:
			return self.__extract_feature(self, imageset)
		img = imageset[0,:,:,:]
		tst = self.__extract_feature(img)
		features = np.zeros((shape[0], *tst.shape), dtype=self.__dtype)
		for i in range(shape[0]):
			img = imageset[i,:,:,:]
			features[i,:] = self.__extract_feature(img)
		return features



