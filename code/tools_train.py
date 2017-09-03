import tools_net as tnet
import tools_tflow as tflw
import tools_data as tdat
import detection_feature as df


def run_network_test():
	data_train = tdat.DataFrame(shape=(64, 64, 3))
	tflw.calc_test(data_train)
	data_deploy = tdat.DataFrame(shape=(1280, 266, 3))
	tflw.calc_test(data_deploy)


def run_training_pass(datafiles):
	data_trn, data_vld, data_tst = tdat.make_dataset(datafiles, df.FeatureExtractorPassthru(), frac_valid=0.10, scaler=tdat.rescale_channel)
	tdat.data_summary(data_trn, data_vld, data_tst)
	tflw.run_test_pass(data_trn, data_vld)


def run_training_search(datafiles):
	data_trn, data_vld, data_tst = tdat.make_dataset(datafiles, df.FeatureExtractorPassthru(), frac_valid=0.10, scaler=tdat.rescale_channel)
	tdat.data_summary(data_trn, data_vld, data_tst)
	stats_file = tflw.make_statsfile()
	tflw.hypersearch(data_trn, data_vld, stats_file)


def run_model_test(datafiles, dump_path="./checkpoint/model.cpkt"):
	data_trn, data_vld, data_tst = tdat.make_dataset(datafiles, df.FeatureExtractorPassthru(), frac_valid=0.10, scaler=tdat.rescale_channel)
	loss, accr = tnet.test_model(tflw.make_measures_xe, data_tst, dump_path=dump_path)
	print(tnet.message_test.format(loss, accr))


if __name__ == "__main__":
	#run_network_test()
	data_files = ["./detection.p", "./detection_extra.p"]
	#run_training_pass(data_files)
	#run_training_search(data_files)
	run_model_test(data_files, dump_path="./model_heatnet_0/batch_64_learn_0.00014_retain_0.71364_leak_0.08786/model.ckpt")
	pass

