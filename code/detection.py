import calibration
import pipeline
import display
import tools_image
import skimage.io as skio
import os
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg


def create_directory(dirname):
	try:
		os.makedirs(dirname, exist_ok=True)
		return True
	except Exception:
		return False


class TraceDisplay(pipeline.TraceHandler):
	"""
	Simple trace handler to display pipeline trace image sequence
	"""
	def setup(self):
		return self

	def run(self, trace):
		print("Showing step trace...")
		for img, lbl in trace:
			display.display_image(img, title=lbl)
		return self


class TraceDump(pipeline.TraceHandler):
	"""
	Simple class to allow dumping pipeline traces to a directory
	"""
	def __init__(self, dir_target, fileext="jpg", full=True, indexed=True):
		self.__target = dir_target
		self.__extention = fileext
		self.__full = full
		self.__indexed = indexed
		self.__index = -1

	def __setup(self):
		assert create_directory(self.__target) == True, "Cannot create target directory"

	def __make_path(self, base, step=None):
		rt = "tr_"
		if self.__indexed:
			rt += "{0:04}_".format(self.__index)
		if step is not None:
			rt += "{0:02}_".format(step)
		rt += base
		pth = os.path.join(self.__target, rt) 
		return pth + os.path.extsep + self.__extention

	def __make_cmap(self, image):
		if tools_image.count_channel(image) == 1:
			return "gray"
		else:
			return None

	def setup(self):
		self.__setup()
		return self

	def run(self, trace):
		self.__index += 1							# increment index
		if self.__full:
			for i, (img, lbl) in enumerate(trace):
				fn = self.__make_path(lbl, i)
				cmap = self.__make_cmap(img)
				mpimg.imsave(fn, img, cmap=cmap)	# skimage doesn't seem to do grayscale easily...
		elif len(trace) > 1:						# only save first (raw) and last (final)
			for i, (img, lbl) in enumerate([trace[0], trace[-1]]):
				fn = self.__make_path(lbl, i)
				cmap = self.__make_cmap(img)
				mpimg.imsave(fn, img, cmap=cmap)
		elif len(trace) == 1:
			fn = self.__make_path(lbl)
			cmap = self.__make_cmap(trace[0][0])
			mpimg.imsave(fn, trace[0][0], cmap=cmap)
		return self


pipe_config = pipeline.Configuration()
pipe_config.trace = False
pipe_config.trace_handler = TraceDump("./image_trace")
pipe_config.enable_lane = False
pipe_main = pipeline.Pipeline(pipe_config)


def run_image_test(dir_source, dir_target):
	files = os.listdir(dir_source)
	print("Processing directory {}, containing {} files...".format(dir_source, len(files)))
	if create_directory(dir_target):
		print("Output directory: {}".format(dir_target))
		for i, fn in enumerate(files):
			print("Processing file {} = {}".format(i+1, fn))
			file_in = os.path.join(dir_source, fn)
			img_source = skio.imread(file_in)
			img_result = pipe_main.process(img_source)
			file_out = os.path.join(dir_target, fn)
			skio.imsave(file_out, img_result)
			print("Result saved for {}".format(fn))
			pipe_main.reset_pipeline()								# reset pipeline to prevent prior images from affecting later ones
	else:
		print("Error creating output directory: {}".format(dir_target))


def run_video_test(dir_source, dir_target):
	files = os.listdir(dir_source)
	print("Processing directory {}, containing {} files...".format(dir_source, len(files)))
	if create_directory(dir_target):
		print("Output directory: {}".format(dir_target))
		for i, fn in enumerate(files):
			print("Processing file {} = {}".format(i+1, fn))
			file_in = os.path.join(dir_source, fn)
			vid_source = VideoFileClip(file_in)
			vid_result = vid_source.fl_image(pipe_main.process)
			file_out = os.path.join(dir_target, fn)
			vid_result.write_videofile(file_out, audio=False)
			pipe_main.reset_pipeline()								# reset pipeline for next video


def run_video_clip(path_source, path_target, clip=None):
	assert clip is not None, "Start and end times for clip are required"
	print("Processing clip from {}, {}...".format(path_source, clip))
	vid_source = VideoFileClip(path_source)
	vid_clip = vid_source.subclip(clip[0], clip[1])
	vid_result = vid_clip.fl_image(pipe_main.process)
	vid_result.write_videofile(path_target, audio=False)


def run_calibration_test():
	fileglob = "./camera_cal/calibration*.jpg"
	filesave = "./calibration.p"
	camera = calibration.calibration_retrieve(filesave, fileglob)
	test = skio.imread("./camera_cal/calibration1.jpg")
	test_corr = camera.apply_correction(test)
	display.display_images(test, test_corr, titles=("original", "corrected"))


if __name__ == "__main__":
	#run_calibration_test()
	#run_image_test("../test_images", "../test_image_out")
	#run_video_test("../test_videos", "../test_video_out")
	#run_video_clip("../test_videos/project_video.mp4", "../test_video_out/project_video.mp4", clip=(4.5,7))

