import numpy as np
import tensorflow as tf
import tools_net as dtt


def make_heatnet_maxpool(input, rate_retain, rate_leak, norm_switch, print_shape=False):
	initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.05)
	layers = [input]
	with tf.name_scope("conv_maxpool"):
		conv_0 = tf.layers.conv2d(input, 128, [3, 3], [1, 1], padding="same", kernel_initializer=initializer)
		conv_0 = tf.layers.batch_normalization(conv_0, training=norm_switch)
		conv_0 = tf.maximum(tf.scalar_mul(rate_leak, conv_0), conv_0)
		conv_0 = tf.nn.dropout(conv_0, rate_retain, name="conv_0")
		layers.append(conv_0)
		conv_1 = tf.layers.conv2d(conv_0, 128, [3, 3], [1, 1], padding="same", kernel_initializer=initializer)
		conv_1 = tf.layers.batch_normalization(conv_1, training=norm_switch)
		conv_1 = tf.maximum(tf.scalar_mul(rate_leak, conv_1), conv_1)
		conv_1 = tf.nn.dropout(conv_1, rate_retain, name="conv_1")
		layers.append(conv_1)
		conv_2 = tf.layers.conv2d(conv_1, 128, [3, 3], [1, 1], padding="same", kernel_initializer=initializer)
		conv_2 = tf.layers.batch_normalization(conv_2, training=norm_switch)
		conv_2 = tf.maximum(tf.scalar_mul(rate_leak, conv_2), conv_2, name="conv_2")
		layers.append(conv_2)
		pool_0 = tf.layers.max_pooling2d(conv_2, [8, 8], [8, 8])
		pool_0 = tf.nn.dropout(pool_0, rate_retain, name="pool_0")
		layers.append(pool_0)
		conv_3 = tf.layers.conv2d(pool_0, 128, [8, 8], [1, 1], padding="valid", kernel_initializer=initializer)
		conv_3 = tf.maximum(tf.scalar_mul(rate_leak, conv_3), conv_3)
		conv_3 = tf.nn.dropout(conv_3, rate_retain, name="conv_3")
		layers.append(conv_3)
		logit = tf.layers.conv2d(conv_3, 1, [1, 1], padding="valid", name="logit", kernel_initializer=initializer)
		layers.append(logit)
	if print_shape:
		for lyr in layers:
			print("layer: {} shape = {}".format(lyr.name, lyr.shape.as_list()))
	return logit


def make_heatnet_conv(input, rate_retain, rate_leak, norm_switch, print_shape=False):
	initializer = tf.contrib.layers.xavier_initializer_conv2d()
	layers = [input]
	with tf.name_scope("fully_conv"):
		conv_0 = tf.layers.conv2d(input, 64, 5, strides=2, activation=None, padding="same", kernel_initializer=initializer)
		conv_0 = tf.layers.batch_normalization(conv_0, training=norm_switch)
		conv_0 = tf.maximum(tf.scalar_mul(rate_leak, conv_0), conv_0, name="conv_0")
		layers.append(conv_0)
		conv_1 = tf.layers.conv2d(conv_0, 128, 5, strides=2, activation=None, padding="same", kernel_initializer=initializer)
		conv_1 = tf.layers.batch_normalization(conv_1, training=norm_switch)
		conv_1 = tf.maximum(tf.scalar_mul(rate_leak, conv_1), conv_1)
		conv_1 = tf.nn.dropout(conv_1, rate_retain, name="conv_1")
		layers.append(conv_1)
		conv_2 = tf.layers.conv2d(conv_1, 64, 5, strides=2, activation=None, padding="same", kernel_initializer=initializer)
		conv_2 = tf.layers.batch_normalization(conv_2, training=norm_switch)
		conv_2 = tf.maximum(tf.scalar_mul(rate_leak, conv_2), conv_2)
		conv_2 = tf.nn.dropout(conv_2, rate_retain, name="conv_2")
		layers.append(conv_2)
		conv_3 = tf.layers.conv2d(conv_2, 32, 3, strides=2, activation=None, padding="same", kernel_initializer=initializer)
		conv_3 = tf.layers.batch_normalization(conv_3, training=norm_switch)
		conv_3 = tf.maximum(tf.scalar_mul(rate_leak, conv_3), conv_3)
		conv_3 = tf.nn.dropout(conv_3, rate_retain, name="conv_3")
		layers.append(conv_3)
		conv_4 = tf.layers.conv2d(conv_3, 8, 3, strides=2, activation=None, padding="same", kernel_initializer=initializer)
		conv_4 = tf.maximum(tf.scalar_mul(rate_leak, conv_4), conv_4, name="conv_4")
		layers.append(conv_4)
		logit = tf.layers.conv2d(conv_4, 1, 1, strides=2, activation=None, padding="valid", kernel_initializer=initializer, name="logit")
		layers.append(logit)
	if print_shape:
		for lyr in layers:
			print("layer: {} shape = {}".format(lyr.name, lyr.shape.as_list()))
	return logit


def make_heatnet_conv2(input, rate_retain, rate_leak, norm_switch, print_shape=False):
	initializer = tf.contrib.layers.xavier_initializer_conv2d()
	layers = [input]
	with tf.name_scope("fully_conv"):
		conv_0 = tf.layers.conv2d(input, 64, 5, strides=1, activation=None, padding="same", kernel_initializer=initializer)
		conv_0 = tf.layers.batch_normalization(conv_0, training=norm_switch)
		conv_0 = tf.maximum(tf.scalar_mul(rate_leak, conv_0), conv_0, name="conv_0")
		layers.append(conv_0)
		conv_1 = tf.layers.conv2d(conv_0, 128, 5, strides=1, activation=None, padding="same", kernel_initializer=initializer)
		conv_1 = tf.layers.batch_normalization(conv_1, training=norm_switch)
		conv_1 = tf.maximum(tf.scalar_mul(rate_leak, conv_1), conv_1)
		conv_1 = tf.nn.dropout(conv_1, rate_retain, name="conv_1")
		layers.append(conv_1)
		conv_2 = tf.layers.conv2d(conv_1, 64, 5, strides=1, activation=None, padding="same", kernel_initializer=initializer)
		conv_2 = tf.layers.batch_normalization(conv_2, training=norm_switch)
		conv_2 = tf.maximum(tf.scalar_mul(rate_leak, conv_2), conv_2)
		conv_2 = tf.nn.dropout(conv_2, rate_retain, name="conv_2")
		layers.append(conv_2)
		conv_3 = tf.layers.conv2d(conv_2, 32, 3, strides=1, activation=None, padding="same", kernel_initializer=initializer)
		conv_3 = tf.layers.batch_normalization(conv_3, training=norm_switch)
		conv_3 = tf.maximum(tf.scalar_mul(rate_leak, conv_3), conv_3)
		conv_3 = tf.nn.dropout(conv_3, rate_retain, name="conv_3")
		layers.append(conv_3)
		conv_4 = tf.layers.conv2d(conv_3, 8, 7, strides=8, activation=None, padding="same", kernel_initializer=initializer)
		conv_4 = tf.maximum(tf.scalar_mul(rate_leak, conv_4), conv_4, name="conv_4")
		layers.append(conv_4)
	logit = tf.layers.conv2d(conv_4, 1, 8, strides=1, activation=None, padding="valid", kernel_initializer=initializer, name="logit")
	layers.append(logit)
	if print_shape:
		for lyr in layers:
			print("layer: {} shape = {}".format(lyr.name, lyr.shape.as_list()))
	return logit


def make_placeholders(data):
	"""
	Create placeholders necessary for network training
	param: data: the data set or suitable substitute
	return: tuple(input, label, rate_retain, norm_switch)
	"""
	data_shape = data.feature_shape()
	with tf.name_scope("input"):
		input = tf.placeholder(tf.float32, [None, *data_shape], name="image")
		label = tf.placeholder(tf.int32, [None], name="label")
		val_unit = tf.constant(1.0, tf.float32)
		rate_retain = tf.placeholder_with_default(val_unit, val_unit.shape, name="rate_retain")
		val_false = tf.constant(False, tf.bool)
		norm_switch = tf.placeholder_with_default(val_false, val_false.shape, name="norm_switch")
	return input, label, rate_retain, norm_switch


def make_measures_mse(logit, label):
	def binary_crossentropy(logit, label):
		eps = np.spacing(1.0)
		label_sig = tf.nn.sigmoid(logit)
		term_a = tf.multiply(label, tf.log(label_sig + eps))
		term_b = tf.multiply(tf.subtract(1.0, label), tf.log(tf.subtract(1.0, label_sig) + eps))
		loss = tf.negative(tf.add(term_a, term_b))
		return loss
	n_logit = logit.shape.as_list()
	logit_flat = tf.contrib.layers.flatten(logit)
	with tf.name_scope("measure"):
		label_sig = tf.nn.sigmoid(logit_flat)
		measure_loss = tf.reduce_mean(tf.squared_difference(label, label_sig))
		predict_test = tf.equal(tf.cast(tf.greater(label_sig, 0.5), tf.float32), label, name="test")
		measure_accuracy = tf.reduce_mean(tf.cast(predict_test, tf.float32), name="accuracy")
	return measure_loss, measure_accuracy


def make_measures_xe(logit, label):
	logit_shape = logit.shape.as_list()
	logit_flat = tf.contrib.layers.flatten(logit)
	logit_flat_shape = logit_flat.shape.as_list()
	label = tf.reshape(label, [-1, 1])
	with tf.name_scope("measure"):
		measure_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_flat, labels=tf.cast(label, tf.float32)))
		predict_test = tf.equal(tf.cast(tf.greater(tf.sigmoid(logit_flat), 0.5), tf.int32), label, name="test")
		measure_accuracy = tf.reduce_mean(tf.cast(predict_test, tf.float32), name="accuracy")
	return measure_loss, measure_accuracy


def make_trainer(measure_loss, rate_learn, decay_rate=0.975, decay_base=10000):
	global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
	rate_control = tf.train.exponential_decay(rate_learn, global_step, decay_base, decay_rate, staircase=True)
	updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(updates):
		trainer = tf.train.AdamOptimizer(rate_control).minimize(measure_loss, global_step=global_step)
	return trainer


def train_model(make_network, data_train, data_valid, n_epoch, size_batch, rate_learn, rate_retain, rate_leak, dump_path="./checkpoint/model.cpkt", n_print=0, accuracy_floor=0.5):
	dtt.reset_dumpdir(dump_path)
	tf.reset_default_graph()
	input, label, retain, norm = make_placeholders(data_train)
	logit = make_network(input, retain, rate_leak, norm)
	meas_loss, meas_accr = make_measures_xe(logit, label)
	saver = tf.train.Saver(max_to_keep=2)
	n_batch = data_train.batch_count(size_batch)
	trainer = make_trainer(meas_loss, rate_learn, decay_base=5*n_batch)
	if n_print <= 0: n_print = max(n_batch//10, 1)
	loss_peak, accr_peak = None, None
	with tf.Session() as TS:
		TS.run(tf.global_variables_initializer())
		term = dtt.Terminator()
		for i in range(n_epoch):
			data_train.shuffle()
			b = 0
			for bx,by,_ in data_train.batch_make(size_batch):
				feed = {
					input: bx,
					label: by,
					retain: rate_retain,
					norm: True
				}
				TS.run(trainer, feed_dict=feed)
				feed.pop(retain)
				feed.pop(norm)
				loss_trn, accr_trn = TS.run([meas_loss, meas_accr], feed_dict=feed)
				b += 1
				if b % n_print == 0:
					print(dtt.message_train.format(i+1, n_epoch, b, n_batch, loss_trn, accr_trn))
			loss_vld, accr_vld = dtt.evaluate(TS, meas_loss, meas_accr, data_valid, size_batch)
			print(dtt.message_valid.format(i+1, n_epoch, loss_vld, accr_vld))
			if term.terminate(accr_vld): break
			if accr_vld >= accuracy_floor:
				if term.new_best():
					loss_peak, accr_peak = loss_vld, accr_vld
					save_path = saver.save(TS, dump_path, global_step=term.current_step())
					print("Model saved to {}".format(save_path))
		return loss_peak, accr_peak


def run_test_pass(data_train, data_valid):
	n_epoch = 100
	size_batch = 64
	rate_learn = 0.01
	rate_retain = 0.5
	rate_leak = 0.1
	loss, accr = train_model(make_heatnet_conv2, data_train, data_valid, n_epoch, size_batch, rate_learn, rate_retain, rate_leak)



stats_file = "./model_heatnet.csv"
path_format = "./model_heatnet_0/batch_{}_learn_{:>3.5f}_retain_{:>3.5f}_leak_{:>3.5f}/model.ckpt"

def make_statsfile(filename=None):
	if filename is None: filename = stats_file
	return dtt.StatisticsFile(filename, ["n_batch", "rate_learn", "rate_retain", "rate_leak", "dump_path", "loss_valid", "accr_valid"])


def search_pass(data_train, data_valid, parameters, n_epoch=100, n_print=0, accuracy_floor=0.5):
	n_batch = parameters[0]
	r_learn = round(parameters[1], 5)
	r_retain = round(parameters[2], 5)
	r_leak = round(parameters[3], 5)
	dump_path = path_format.format(n_batch, r_learn, r_retain, r_leak)
	loss, accr = train_model(make_heatnet_conv2, data_train, data_valid, n_epoch, n_batch, r_learn, r_retain, r_leak, dump_path=dump_path, n_print=n_print, accuracy_floor=accuracy_floor)
	return loss, accr, dump_path


def hypersearch(data_train, data_valid, stats_file):
	accr_floor = 0.80
	np.random.seed(4357)
	batch = [16, 32, 64]
	learn = np.linspace(-np.log(0.1), -np.log(.0001), 20)
	for i in range(25):
		n_epoch = 100
		n_batch = np.random.choice(batch)
		r_learn = np.exp(-np.random.choice(learn))
		r_retain = np.random.uniform(low=0.40, high=1.0)
		r_leak = np.random.uniform(low=0.0, high=0.40)
		loss, accr, path = search_pass(data_train, data_valid, [n_batch, r_learn, r_retain, r_leak], n_epoch, accuracy_floor=accr_floor)
		stats_file.write([n_batch, r_learn, r_retain, r_leak, path, loss, accr])


def make_predictor(data, rate_leak):
	input, label, retain, norm = make_placeholders(data)
	convnet = make_heatnet_conv2(input, retain, rate_leak, norm)
	predict = tf.sigmoid(convnet, name="predictor")
	return predict


def calc_test(data):
	#input, label, retain, norm, = make_placeholders(data)
	#poolnet = make_heatnet_maxpool(input, retain, 0.0, norm, print_shape=True)
	#poolnet_shp = poolnet.shape.as_list()
	tf.reset_default_graph()
	#print("")
	input, label, retain, norm, = make_placeholders(data)
	convnet = make_heatnet_conv2(input, retain, 0.0, norm, print_shape=True)
	convnet_shp = convnet.shape.as_list()
	#test_in = tf.placeholder(tf.float32, [None, 1, 1, 1])
	#test_val = 2*np.ones((1, 1, 1, 1), dtype=np.float32)
	#test_out = make_spreadmap(test_in)
	#shape_test_out = test_out.shape.as_list()
	#with tf.Session() as TS:
	#	TS.run(tf.global_variables_initializer())
	#	result = TS.run(test_out, feed_dict={test_in: test_val})
	#	pass
