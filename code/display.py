import matplotlib.pyplot as plt


def display_images(lefthand, righthand, titles=(None, None), asgray=(False, False)):
	"""
	Display two images side-by-side
	param: lefthand: the left-hand image
	param: righthand: the right-hand image
	param: titles: 2-tuple of string image titles
	param: asgray: 2-tuple of boolean grayscale flags to control image dislay
	return: None
	"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40,20))
	ax1.imshow(lefthand) if asgray[0] == False else ax1.imshow(lefthand, cmap="gray")
	if titles[0]: ax1.set_title(titles[0], fontsize=25)
	ax2.imshow(righthand) if asgray[1] == False else ax2.imshow(righthand, cmap="gray")
	if titles[1]: ax2.set_title(titles[1], fontsize=25)
	plt.show()


def display_image(image, title=None, asgray="auto"):
	"""
	Simple image dislay function
	param: image: the image to be displayed
	param: title: the title to be given (optional)
	param: asgray: whether the image should be should in grayscale or heatmap (optional, default = auto)
					if auto, then if the number of color channels is one, grayscale is selected
	return: None
	"""
	as_gray = False
	cmap_name = "gray"
	if asgray == "auto":
		shape = image.shape
		if len(shape) < 3 or shape[-1] < 2:
			as_gray = True
	elif asgray == "gray":
		as_gray = True
	elif asgray == "heat":
		as_gray = True
		cmap_name = "heat"
	fig, ax = plt.subplots(1, 1, figsize=(30,20))
	ax.imshow(image) if as_gray == False else ax.imshow(image, cmap=cmap_name)
	if title: ax.set_title(title, fontsize=25)
	plt.show()


def display_image_grid(image_set, title=None, subtitle=None, ncol=4, figsize=(12,20)):
	"""
	Display a grid of images
	param: image_set: the images to be displayed
	param: title: Title for the plot (optional)
	param: subtitle: Subtitles for each subplot (optional)
	param: ncol: the number of columns to arrange (optional, default = 4)
	param: figsize: the size of the figure (optional)
	return: nothing
	"""
	n = len(image_set)
	nrow = n//ncol
	if n % ncol != 0: nrow += 1
	fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize)
	fig.tight_layout()
	if title:
		fig.suptitle(title, fontsize=20, y=1.0)
	has_subtitles = bool((subtitle is not None) and len(subtitle) == n)
	idx = 0
	for i in range(nrow):
		for j in range(ncol):
			axes[i][j].set_axis_off()
			if idx >= n: continue
			axes[i][j].imshow(image_set[idx])
			if has_subtitles:
				axes[i][j].set_title(subtitle[idx])
			idx += 1
	plt.show()