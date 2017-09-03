from pathlib import Path
import shutil
import numpy as np
import tensorflow as tf


class Terminator(object):
	def __init__(self, relax=10, mode="ascend"):
		self.__mode_ascend = bool(mode=="ascend")
		self.__relax = int(relax)
		self.__count = int(relax)
		self.__value = None
		self.__viter = None
		self.__iter = int(0)

	def __incr(self):
		self.__iter += 1

	def __ascend(self, value):
		if value > self.__value:
			self.__count = self.__relax
			self.__value = value
			self.__viter = self.__iter
		else:
			self.__count -= 1
			if self.__count == 0: return False
		return True

	def __descend(self, value):
		if value < self.__value:
			self.__count = self.__relax
			self.__value = value
			self.__viter = self.__iter
		else:
			self.__count -= 1
			if self.__count == 0: return False

	def reset(self):
		self.__value = None
		self.__viter = None
		self.__iter = 0
		self.__count = self.__relax

	def new_best(self):
		if self.__value:
			return bool(self.__viter == self.__iter)
		return False

	def current_step(self):
		return self.__viter

	def maintain(self, value):
		self.__incr()
		if self.__value is None:
			self.__value = value
			return True
		if self.__mode_ascend:
			return self.__ascend(value)
		else:
			return self.__descend(value)

	def terminate(self, value):
		return not self.maintain(value)


class StatisticsFile(object):
	def __init__(self, filename, fields):
		assert filename is not None, "invalid filename"
		assert len(fields) > 0, "no fields specified"
		self.__filename = filename
		self.__fields = fields
		self.__header = ",".join(fields)
		self.__count = len(fields)
		try:
			self.__file = open(filename, "a")
			self.__file.write(self.__header + "\n")
		except Exception:
			return
		self.__format = ""
		for fld in self.__fields:
			if fld.startswith("rate"):
				self.__format += "{:>3.5f},"
			else:
				self.__format += "{},"
		self.__format = self.__format[:-1] + "\n"

	def fields(self):
		return self.__fields

	def write(self, values):
		if len(values) == self.__count:
			self.__file.write(self.__format.format(*values))
			self.__file.flush()			# make sure it gets written to disk
		return self


class ModelRunner(object):
	def __init__(self, model_path="./checkpoint/model.ckpt", graph_call=None):
		modeldir = dump_directory(model_path)
		checkpoint = tf.train.latest_checkpoint(modeldir)
		tf.reset_default_graph()
		if graph_call is not None: graph_call()		# build computational graph
		self.__TS = tf.Session()
		if graph_call is None:
			loader = tf.train.import_meta_graph(checkpoint + ".meta")	# load graph
			loader.restore(self.__TS, checkpoint)
		else:
			loader = tf.train.Saver()
			loader.restore(self.__TS, checkpoint)
		g = self.__TS.graph
		self.__input = g.get_tensor_by_name("input/image:0")
		self.__predict = g.get_tensor_by_name("predictor:0")

	def predict(self, image):
		return self.__TS.run(self.__predict, feed_dict={self.__input: image})


def reset_dumpdir(dumppath):
	parent = Path(dumppath).parent
	if parent.exists():
		try:
			shutil.rmtree(parent.name)
		except Exception:
			pass
	try:
		parent.mkdir(parents=True, exist_ok=True)
	except Exception:
		pass


def dump_directory(dumppath):
	dir = Path(dumppath)
	if dir.is_dir():
		return dumppath
	else:
		return str(Path(dumppath).parent.relative_to("."))


message_train = "Epoch {:>3}/{:>3} - Batch {:>3}/{:>3} | Training  : loss = {:>3.5f}, accuracy = {:>3.5f}"
message_valid = "Epoch {:>3}/{:>3} | Validation: loss = {:>3.5f}, accuracy = {:>3.5f}"
message_test = "Testing: loss = {:>3.5f}, accuracy = {:>3.5f}"


def make_measures(logit, label):
	logit_shape = logit.shape.as_list()
	logit_flat = tf.contrib.layers.flatten(logit)
	logit_flat_shape = logit_flat.shape.as_list()
	label = tf.reshape(label, [-1, 1])
	with tf.name_scope("measure"):
		measure_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_flat, labels=tf.cast(label, tf.float32)))
		predict_test = tf.equal(tf.cast(tf.greater(tf.sigmoid(logit_flat), 0.5), tf.int32), label, name="test")
		measure_accuracy = tf.reduce_mean(tf.cast(predict_test, tf.float32), name="accuracy")
	return measure_loss, measure_accuracy


def evaluate(session, loss, accuracy, data, size_batch=128):
	g = session.graph
	input = g.get_tensor_by_name("input/image:0")
	label = g.get_tensor_by_name("input/label:0")
	loss_t = 0.0
	accu_t = 0.0
	for bx,by,n in data.batch_make(size_batch):
		loss_v, accr_v = session.run([loss, accuracy], feed_dict={input:bx, label: by})
		loss_t += loss_v*n
		accu_t += accr_v*n
	return loss_t/data.length, accu_t/data.length


def test_model(make_measure, data, dump_path="./checkpoint/model.ckpt"):
	dumpdir = dump_directory(dump_path)
	checkpoint = tf.train.latest_checkpoint(dumpdir)
	tf.reset_default_graph()
	with tf.Session() as TS:
		loader = tf.train.import_meta_graph(checkpoint + ".meta")
		loader.restore(TS, checkpoint)
		g = TS.graph
		logit = g.get_tensor_by_name("logit/BiasAdd:0")
		label = g.get_tensor_by_name("input/label:0")
		measure_loss, measure_accr = make_measure(logit, label)
		loss, accr = evaluate(TS, measure_loss, measure_accr, data)
	return loss, accr