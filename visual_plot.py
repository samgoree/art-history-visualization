"""
Create a scatterplot of images with matplotlib and PIL
"""

import numpy as np
import matplotlib.pyplot as plt

import changepoint_estimation


def plot_images(X, Y, image_paths, canvas_shape=(2000,2000,3), 
	target_image_resolution=75, ax=None, background_color=(1,1,1)):
	"""
	Plot several images on a matplotlib axis
	X, Y: the coordinates for the scatterplot
	image_paths: a list of the same length as X and Y with paths to images
	canvas_shape: the dimensions (in pixels) of the canvas images will be drawn
	on. The canvas shape will determine the units for matplotlib, so the same
	canvas shape should be provided to other functions plotting on top of the
	same axes.
	target_image_resolution: Images will be resized to this image height. If
	this value is None, images will be left at their original size/resolution.
	ax: a matplotlib axis to plot on.
	background_color: RGB float values for the canvas background color. (0,0,0)
	is black, (1,1,1) is white.
	"""

	if len(X) != len(Y):
		raise ValueError(
			'X and Y provided are of different lengths. len(X): ' 
			+ str(len(X)) + ' len(Y): ' + str(len(Y))
		)
	if len(X) != len(image_paths):
		raise ValueError(
			'Different number of X values than image_paths: len(X): ' 
			+ str(len(X)) + ' len(image_paths): ' + len(image_paths)
		)
    
    canvas = np.zeros(canvas_shape) + background_color
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(18,18))

    Y_max = np.max(Y)
    Y_min = np.min(Y)
    Y_range = Y_max - Y_min
    
    X_max = np.max(X)
    X_min = np.min(X)
    X_range = X_max - X_min
    
    def x_coord(x):
        return int((x - X_min) * (canvas.shape[1] - target_image_resolution) / X_range)
    def y_coord(y):
        return int((y - Y_min) * (canvas.shape[0] - target_image_resolution) / Y_range)

    for i in range(len(Y)):
        try:
            im = cv2.imread(image_paths[i])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im 
        except Exception as e:
            print(e)
            print(image_paths[i])
            print(im)
            continue
        x = x_coord(X[i])
        y = canvas.shape[0] - y_coord(Y[i])
        
        downsample_factor = int(max(im.shape[0], im.shape[1]) / target_image_resolution)
        im = im[::downsample_factor,::downsample_factor]

        bound_x = int(min(x+im.shape[1], canvas.shape[1]))
        bound_y = int(min(y+im.shape[0], canvas.shape[0]))

        # handle color to grayscale
        if (len(canvas.shape) < 3 or canvas.shape[2] == 1) and (len(im.shape) == 3 and im.shape[2] > 1):
        	im_to_draw = np.mean(im[:bound_y - int(y),:bound_x - int(x)], axis=2)
        else:
        	im_to_draw = im[:bound_y - int(y),:bound_x - int(x)]

        # handle uint8 images
        if np.max(im_to_draw) > 1:
        	im_to_draw /= 256


        canvas[int(y):bound_y,int(x):bound_x] = im_to_draw
    
    plt.imshow(canvas[::-1], origin='lower')
    return ax
    

def plot_means(X, Y, canvas_shape=(2000,2000,3), window_size=1, ax=None, error_bars=True):
	"""
	Plot the mean of Y for each value of X (plus or minus window_size) and align
	it to the canvas
	X, Y: the coordinates for the scatterplot
	canvas_shape: the canvas shape used for visual_plot
	window_size: the mean at position x will be taken for (X,Y) pairs where 
	x - window_size < X < x + window_size
	error_bars: use error bars to show standard error
	"""


	if ax is None:
        fig, ax = plt.subplots(figsize=(18,18))
    avg = []
    stderr = []
    nonzero_years = []
    for x in np.arange(X_min, X_max):
        mask = (x - window_size <= X < x + window_size)
        if np.sum(mask) > 0:
            nonzero_years.append(year)
            avg.append(np.mean(np.array(Y)[mask]))
            stderr.append(np.std(np.array(Y)[mask]) / np.sqrt(np.sum(mask)))
    if error_bars:
    	ax.errorbar(
    		[x_coord(y) for y in nonzero_years], 
    		[y_coord(a) for a in avg], 
    		yerr=[s *  canvas.shape[0] / Y_range for s in stderr], color='k'
    	)
    else:
    	ax.plot([x_coord(y) for y in nonzero_years], [y_coord(a) for a in avg], color='k')
    return ax


def count_params(n_changepoints, param_fcn):
    if model == 'linear':
        # two parameters per line, n_changepoints + 1 lines, plus n_changepoints
        # changepoints, which are themselves parameters
        return (n_changepoints + 1) * 2
    elif model == 'constant':
        # one parameter for each of n_changepoints + 1 constant sections, 
        # plus n_changepoints
        return (n_changepoints + 1) * 1 + n_changepoints


def plot_periodization(X, Y, canvas_shape=(2000,2000,3), 
    min_changepoints=0, max_changepoints=10, 
    model='linear', n_params='linear', ax=None):
    """
    Fit a piecewise model to the data using changepoint_estimation.find_changepoints and plot
    X and Y: the coordinates for the scatterplots
    canvas_shape: the canvas shape used for visual plot
    min_changepoints, max_changepoints: the bounds on the search space for the 
    change points. More data will usually require more change points
    model: model for changepoint_estimation.find_changepoints, either 'linear'
    'constant' or callable which takes x and y and returns model training error
    param_fcn: either 'linear', 'constant' (same as model) or the number of 
    parameters in the model

    Note this function makes use of the Bayesian information criterion as a
    regularizer (to avoid overfitting and choosing max_changepoints each time)
    This works well in practice, but isn't a perfect heuristic.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(18,18))

    # find a periodization
    mask = ~np.isnan(X) & ~np.isnan(Y)
    error_vals = []
    changepoint_lists = []
    for i in range(min_changepoints, max_changepoints):
        changepoints, error = changepoint_estimation.find_changepoints(X[mask], Y[mask], i, model=model)
        changepoint_lists.append(changepoints)
        error_vals.append(error)
    if n_params == 'linear': n_params = 2
    elif n_params == 'constant': n_params = 1
    total_params = np.arange(min_changepoints, max_changepoints) * (n_params + 1) + 1

    bic = BIC(total_params, np.sum(mask), error_vals)
    
    changepoints = changepoint_lists[np.argmin(bic)]
    
    changepoints = [np.nanmin(years)] + changepoints + [np.nanmax(years)]
    

    for i in range(1, len(changepoints)):
        segment_mask = np.less(years, changepoints[i]) & np.greater_equal(years, changepoints[i-1]) & mask
        if np.sum(segment_mask) == 0:
            continue
        # m,b,r,p,_ = scipy.stats.linregress(years[segment_mask], np.array(Y)[segment_mask])
        b = np.mean(np.array(Y)[segment_mask])
        x_vals = np.arange(changepoints[i-1], changepoints[i])
        ax.plot([x_coord(x) for x in x_vals], [y_coord(y) for y in np.ones_like(x_vals) * b], color='b')
    


def visual_plot(X, Y, image_paths):
    # TODO

    plt.ylabel(ylabel)
    plt.yticks(
        [y_coord(y) for y in np.arange(
            Y_min * 0.9, Y_max * 1.1, Y_range/10
        )], [round(s,2) for s in np.arange(Y_min * 0.9, Y_max * 1.1, Y_range/10)]
    )
    if X_max - X_min < 100:
        plt.xticks([x_coord(x) for x in np.arange(X_min - 2, X_max + 2, 2)], np.arange(X_min - 2, X_max + 2, 2, dtype=int))
    else:
        plt.xticks([x_coord(x) for x in np.arange(X_min - 5, X_max + 5, 5)], np.arange(X_min - 5, X_max + 5, 5, dtype=int))