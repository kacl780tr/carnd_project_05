import numpy as np
from sklearn import svm
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import pickle
import detection_feature as df


def model_dump(filename, model, extractor=None):
	path = Path(filename)
	if path.exists():
		path.unlink()
	model_dict = {
		"model": model
		}
	if extractor is not None:
		model_dict["extractor"] = extractor
	try:
		with open(filename, mode="wb") as tgt:
			pickle.dump(model_dict, tgt)
	except pickle.PicklingError:
		model_dump(filename, model)
	except Exception:
		pass


def model_load(filename):
	path = Path(filename)
	if path.exists():
		with open(filename, mode="rb") as src:
			try:
				model_dict = pickle.load(src)
				if len(model_dict) == 2:
					return model_dict["model"], model_dict["extractor"]
				else:
					return model_dict["model"], None
			except Exception:
				pass
	else:
		return None, None


def run_search_pass(data_train, parameter_grid, n_fold=5, n_jobs=8, verbose=2, n_iter=0):
	template = svm.SVC()
	search = None
	if n_iter == 0:
		search = GridSearchCV(template, parameter_grid, cv=n_fold, n_jobs=n_jobs, verbose=verbose)
	else:
		search = RandomizedSearchCV(template, parameter_grid, cv=n_fold, n_jobs=n_jobs, verbose=verbose, n_iter=n_iter)
	search.fit(data_train.feature, data_train.label)
	return search.best_estimator_


def run_evaluation(data_test, model):
	assert data_test is not None and data_test.length > 0, "No test data provided"
	try:
		predict = model.predict(data_test.feature)
		accr = accuracy_score(data_test.label, predict)
		score = f1_score(data_test.label, predict)
		return accr, score
	except Exception:
		return -1, -1


def make_parameter_grid(rangelist):
	grid = {}
	for rng in rangelist:
		p_name = rng[0]
		p_list = np.linspace(rng[1], rng[2], rng[3]).tolist()
		grid[p_name] = p_list
	return grid


