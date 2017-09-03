import numpy as np

def make_focus_region(shape, alpha, beta, gamma):
	"""
	Create a focus region for the given image shape and supplied parameters
	param: shape: the shape of the target image
	param: alpha: the horizontal centered fraction of the target image included at upper boundary
	param: beta: the location of the upper boundary as a fraction of image height
	param: gamma: the horizontal centered fraction of the target image included at lower boundary (bottom of image)
	return: a [4 x 2] array of vertices in lower -> upper left -> upper -> lower right
	"""
	assert len(shape) >= 2
	sx, sy = shape[1], shape[0]
	region_source = []
	region_source.append([int(sx*0.5*(1.0 - gamma)), sy])		# lower left
	region_source.append([int(sx*0.5*(1.0 - alpha)), sy*beta])	# upper left
	region_source.append([int(sx*0.5*(1.0 + alpha)), sy*beta])	# upper right
	region_source.append([int(sx*0.5*(1.0 + gamma)), sy])		# lower right
	return np.array(region_source, dtype=np.float32)


def make_target_region(shape, region_source, beta, delta):
	"""
	Create a target region for the perspective transformation given image shape, source region, and supplied parameters
	param: shape: the shape of the target image
	param: region_source: the region to which perspective transformation will be applied
	param: beta: the location of the upper boundary as a fraction of image height
	param: delta: the mixing ratio for the horizontal coordinates
	return: a [4 x 2] array of vertices in lower -> upper left -> upper -> lower right
	"""
	assert len(shape) >= 2
	assert region_source.shape[0] == 4
	sy = shape[0]
	tgt_top = beta*sy
	tgt_left = delta*region_source[0][0] + (1.0 - delta)*region_source[1][0]
	tgt_rght = delta*region_source[3][0] + (1.0 - delta)*region_source[2][0]
	region_target = []
	region_target.append([tgt_left, sy])
	region_target.append([tgt_left, tgt_top])
	region_target.append([tgt_rght, tgt_top])
	region_target.append([tgt_rght, sy])
	return np.array(region_target, dtype=np.float32)
	

def make_focus_regions(shape, alpha, beta, gamma, delta):
	"""
	Create focus and target regions for the perspective transformation given image shape and supplied parameters
	param: shape: the shape of the target image
	param: alpha: the horizontal centered fraction of the target image included at upper boundary
	param: beta: the location of the upper boundary as a fraction of image height
	param: delta: the mixing ratio for the horizontal coordinates
	return: a [4 x 2] array of vertices in lower -> upper left -> upper -> lower right
	"""
	assert len(shape) >= 2
	sx, sy = shape[1], shape[0]
	region_source = []
	region_source.append([int(sx*0.5*(1.0 - gamma)), sy])		# lower left
	region_source.append([int(sx*0.5*(1.0 - alpha)), sy*beta])	# upper left
	region_source.append([int(sx*0.5*(1.0 + alpha)), sy*beta])	# upper right
	region_source.append([int(sx*0.5*(1.0 + gamma)), sy])		# lower right
	tgt_left = int(delta*region_source[0][0] + (1.0 - delta)*region_source[1][0])
	tgt_rght = int(delta*region_source[3][0] + (1.0 - delta)*region_source[2][0])
	region_target = []
	region_target.append([tgt_left, sy])
	region_target.append([tgt_left, 0])
	region_target.append([tgt_rght, 0])
	region_target.append([tgt_rght, sy])
	return np.array(region_source, dtype=np.float32), np.array(region_target, dtype=np.float32)


def make_region_from_lines_flex(shape, lines):
	"""
	Create focus region from provided lines
	param: shape: the shape of the target image
	param: lines: the provided lines if only one line is provided, then region is formed by reflecting line across vertical center
	return: a [4 x 2] array of vertices in lower -> upper left -> upper -> lower right or None if no lines provided
	"""
	if len(lines) == 2:
		region = []
		lhs = lines[0]		# left-hand line
		xb, yb, xe, ye = lines[0][:]
		region.append([xb, yb])
		region.append([xe, ye])
		xb, yb, xe, ye = lines[1][:]
		region.append([xe, ye])
		region.append([xb, yb])
		return np.array(region, dtype=np.float32)
	elif len(lines) == 1:
		sx = shape[1]
		region = []
		xb, yb, xe, ye = lines[0][:]
		region.append([xb, yb])
		region.append([xe, ye])
		region.append([sx - xe, ye])
		region.append([sx - xb, yb])
		return np.array(region, dtype=np.float32)
	else:
		return None


def make_region_from_lines(lines):
	"""
	Create focus region from provided lines
	param: lines: the provided lines (must contain 2 lines)
	return: a [4 x 2] array of vertices in lower -> upper left -> upper -> lower right or None if 2 lines not provided
	"""
	if len(lines) == 2:
		region = []
		lhs = lines[0]		# left-hand line
		xb, yb, xe, ye = lines[0][:]
		region.append([xb, yb])
		region.append([xe, ye])
		xb, yb, xe, ye = lines[1][:]
		region.append([xe, ye])
		region.append([xb, yb])
		return np.array(region, dtype=np.float32)
	return None


def make_vanishing_region(region_source, frac=0.98):
	"""
	Extend region toward vanishing point
	param: region_source: the base region
	param: frac: the fraction of distance to vanishing point to extend
	return: a [4 x 2] array of vertices in lower -> upper left -> upper -> lower right
	"""
	assert region_source.shape[0] == 4
	points = []
	for i in range(4):										# form points in homogeneous coordinates
		x, y = region_source[i,:]
		points.append(np.array([x, y, 1.0]))
	lines = []
	for i in range(0, 4, 2):									# compute lines
		lines.append(np.cross(points[i], points[i+1]))
	point_van = np.cross(lines[0], lines[1])				# intersection = vanishing point
	if point_van[2] != 0.0:
		point_van /= point_van[2]							# normalize to image plane
	point_van = point_van[:2]
	point_van *= frac
	region_van = np.copy(region_source)
	region_van[1,:] = (point_van + (1.0 - frac)*region_van[1])
	region_van[2,:] = (point_van + (1.0 - frac)*region_van[2])
	return region_van


def make_linesets(lines, shape):
	"""
	Process the set of lines returned by Hough transform:
		- identify groups
		- extrapolate from image base to upper part of target region
	:param lines: the set of lines identified by Hough transform
	:param shape: the shape of the image the lines were drawn from
	:return: tuple (lines_lhs, lines_rhs)
	"""
	def match_side(div, pointA, pointB):
		match = False
		right = None
		normal_a = np.cross(div[1] - div[0], pointA - div[0])
		normal_b = np.cross(div[1] - div[0], pointB - div[0])
		if np.dot(normal_a, normal_b) >= 0.0:
			nrm = np.linalg.norm(pointB - pointA, ord=2)*np.linalg.norm(div[1] - div[0], ord=2)
			if np.abs(np.dot(div[1] - div[0], pointB - pointA)) < 0.3*nrm:  # check for lines substantially perpendicular to divider
				return match, right
			match = True
			right = (normal_a[2] < 0.0)     # set side based on sign of normal component
		return match, right

	dim_x = shape[1]
	dim_y = shape[0]
	divider = [np.array([dim_x//2, 0.0, 0.0]), np.array([dim_x//2, dim_y, 0.0])]     # define the plane through vertical center of image
	lines_rhs = []
	lines_lhs = []
	if lines is not None:
		for line in lines:
			for x1, y1, x2, y2 in line:
				ln_s = np.array([x1, y1, 0.0])
				ln_e = np.array([x2, y2, 0.0])
				m, r = match_side(divider, ln_s, ln_e)
				if m:
					if r:
						lines_rhs.append(line)
					else:
						lines_lhs.append(line)
	return lines_lhs, lines_rhs


def make_line_blend(group, shape, beta):
	"""
	Blend a group of lines, presumably very colinear to produce an average line
	:param group: a list of 4-element line definitions
	:param shape: the shape of the image
	:param beta: the beta parameter determining the height of focus region
	:return: a single-element list containing the blended line
	"""
	dim_y = shape[0]
	y_lo = dim_y
	y_hi = int(dim_y*beta)
	mean_x_lo = 0.0
	mean_x_hi = 0.0
	n_mean = 0.0
	for line in group:
		for x1, y1, x2, y2 in line:
			den = x2 - x1
			if den == 0:        # vertical line
				mean_x_lo += x2
				mean_x_hi += x1
				n_mean += 1.0
				continue
			slope = float(y2 - y1)/float(den)
			if slope == 0.0:    # horizontal line - skip
				continue
			incpt = 0.5*float(y1 + y2) - slope*0.5*float(x1 + x2)
			mean_x_lo += (y_lo - incpt)/slope
			mean_x_hi += (y_hi - incpt)/slope
			n_mean += 1.0
	if n_mean > 0:
		mean_x_lo /= n_mean
		mean_x_hi /= n_mean
		return [[mean_x_lo, y_lo, mean_x_hi, y_hi]]		# single blended line spanning coverage zone
	return []											# empty list if group empty or only horizontal lines


def verify_region(region, template, tolerance=0.10):
	"""
	Test a region against a known-good template
	param: region: to be verified
	param: template: a (hopefully) known-good template region
	param: tolerance: the allowable departure from the template
	return: True is region passes tests, False otherwise
	"""
	assert region.shape[0] == 4, "Invalid region"
	assert template.shape[0] == 4, "Invalid template"
	for i in range(4):
		dr = 0
		dt = 0
		for j in range(2):
			dr += (region[(i + 1) % 4][j] - region[i][j])**2
			dt += (template[(i + 1) % 4][j] - template[i][j])**2
		dr = np.sqrt(dr)
		dt = np.sqrt(dt)
		if 2.0*np.abs(1.0 - dr/(dt + np.spacing(10.0))) > tolerance: return False	# apply to each line, rather than aggregate
	return True
