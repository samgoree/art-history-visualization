def range_filter(x,y,i,j):
    # find elements of (x,y) where X is between i and j
    mask = np.less(x, j) & np.greater_equal(x,i)
    return x[mask], y[mask]

def linear_model(x, y):
    """
    Measure the sum square error with a linear fit
    """
    results = scipy.stats.linregress(x, y)
    sse = np.sum((x_masked * results[0] + results[1] - y_masked)**2)
    return sse

def constant_model(x, y):
    """
    measure the sum square error with a constant fit
    """
    results = np.mean(y_masked)
    sse = np.sum((results - y_masked)**2)


def apply_model(x, y, model):
    if model == 'linear':
        sse = linear_model(x, y)
    elif model == 'constant':
        sse = constant_model(x, y)
    elif model is callable:
        sse = model(x, y)
    else:
        raise ValueError('model must be "linear_model", "constant_model" or callable')


def find_changepoints(x, y, n_changepoints, model='linear'):
    """
    TODO
    """
    # this is a dynamic programming problem
    # the optimal solution from i to j using n changepoints is the min error from i to k with n-1 changepoints plus the error on k to j with 0
    
    if n_changepoints == 0:
        return [], apply_model(x, y)

    candidates = sorted(set(x))
    dp_table = np.zeros([len(candidates), len(candidates)+1, n_changepoints]) + np.inf
    choices = np.zeros([len(candidates), len(candidates)+1, n_changepoints],dtype=int)
    # base cases
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)+1):
            x_masked, y_masked = range_filter(x,y,candidates[i],candidates[min(j, len(candidates)-1)])
            if len(x_masked) < 2:
                dp_table[i,j,0] = np.inf
            else:
                
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