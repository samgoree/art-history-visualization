import scipy
import numpy as np

def BIC(n_params, n_data, sum_of_squares):
    return n_data * np.log(sum_of_squares/n_data) + n_params * np.log(n_data)

def range_filter(x,y,i,j):
    # find elements of (x,y) where X is between i and j
    mask = np.less(x, j) & np.greater_equal(x,i)
    return x[mask], y[mask]

def score_linear_model(x, y):
    """
    Measure the sum square error with a linear fit
    """
    results = scipy.stats.linregress(x, y)
    sse = np.sum((x * results[0] + results[1] - y)**2)
    if any([np.isnan(r) for r in results]): return np.inf
    return sse

def predict_linear_model(x_train, y_train, x_test):
    results = scipy.stats.linregress(x_train, y_train)
    return x_test * results[0] + results[1]

def score_constant_model(x, y):
    """
    measure the sum square error with a constant fit
    """
    results = np.mean(y)
    sse = np.sum((results - y)**2)
    return sse

def predict_constant_model(x_train, y_train, x_test):
    result = np.mean(y_train)
    return 0 * x_test + result


def score_model(x, y, model):
    if model == 'linear':
        sse = score_linear_model(x, y)
    elif model == 'constant':
        sse = score_constant_model(x, y)
    elif model is callable:
        sse = model(x, y)
    else:
        raise ValueError('model must be "linear_model", "constant_model" or callable')
    return sse

def predict_model(x_train, y_train, x_test, model):
    if model == 'linear':
        return predict_linear_model(x_train, y_train, x_test)
    elif model == 'constant':
        return predict_constant_model(x_train, y_train, x_test)
    elif model is callable:
        return model(x_train, y_train, x_test)
    else:
        raise ValueError('model must be "linear_model", "constant_model" or callable')
    return sse


def find_changepoints(x, y, n_changepoints, model='linear'):
    """
    find the changepoints for a model with n_changepoints fit to data (x,y)
    x,y are data arrays which should be the same length
    n_changepoints should be an integer < len(x)
    model should either be 'linear' or 'constant'
    returns a list of changepoint locations and the sum of squared errors
    """
    # this is a dynamic programming problem
    # the optimal solution from i to j using n changepoints is the min error from i to k with n-1 changepoints plus the error on k to j with 0
    
    assert len(x) == len(y)
    if n_changepoints == 0:
        return [], score_model(x, y, model)

    candidates = sorted(set(x))
    dp_table = np.zeros([len(candidates), len(candidates)+1, n_changepoints]) + np.inf
    choices = np.zeros([len(candidates), len(candidates)+1, n_changepoints],dtype=int)
    # base cases
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)+1):
            x_masked, y_masked = range_filter(x,y,candidates[i],candidates[min(j, len(candidates)-1)])
            if len(x_masked) < 2 or np.min(x_masked) == np.max(x_masked):
                dp_table[i,j,0] = np.inf
            else:
                sse = score_model(x_masked, y_masked, model)
                if np.isnan(sse):
                    dp_table[i,j,0] = np.inf
                else:
                    dp_table[i,j,0] = sse
    # dynamic programming
    for n in range(1, n_changepoints):
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                options = np.arange(i+1,j+1)
                if len(options) == 0:
                    dp_table[i,j,n] = np.inf
                else:
                    errors = dp_table[i,options,n-1] + dp_table[options,j,0]
                    dp_table[i,j,n] = np.min(errors)
                    choices[i,j,n] = np.argmin(errors) + 1
    
    # recover solution
    options = np.arange(1,len(candidates))
    
    optimal_split = np.argmin(dp_table[0,options,-1] + dp_table[options,len(candidates),0])
    sse = np.min(dp_table[0,options,-1] + dp_table[options,len(candidates),0])
    splits = [optimal_split]
    for n in range(n_changepoints-1,0,-1):
        splits.append(choices[0,splits[-1],n])
    return [candidates[s] for s in splits[::-1]], sse