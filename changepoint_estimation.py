import scipy
import numpy as np
from scipy.special import logsumexp

def BIC(n_params, n_data, sum_of_squares):
    return n_data * np.log(sum_of_squares/n_data) + n_params * np.log(n_data)

def BIC_log_likelihood(n_params, n_data, ll):
    return n_params * np.log(n_data) - 2 * ll

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

def score_multivariate_model(x,y):
    results = np.mean(y, axis=0)
    sse = np.sum((results - y)**2)
    return sse

def predict_multivariate_model(x_train, y_train, x_test):
    result = np.mean(y_train, axis=0)
    return result

def score_model(x, y, model):
    if model == 'linear':
        sse = score_linear_model(x, y)
    elif model == 'constant':
        sse = score_constant_model(x, y)
    elif model == 'multivariate':
        sse = score_multivariate_model(x, y)
    elif model is callable:
        sse = model(x, y)
    else:
        raise ValueError('model must be "linear", "constant", "multivariate" or callable')
    return sse

def predict_model(x_train, y_train, x_test, model):
    if model == 'linear':
        return predict_linear_model(x_train, y_train, x_test)
    elif model == 'constant':
        return predict_constant_model(x_train, y_train, x_test)
    elif model == 'multivariate':
        return predict_multivariate_model(x_train, y_train, x_test)
    elif model is callable:
        return model(x_train, y_train, x_test)
    else:
        raise ValueError('model must be "linear", "constant", "multivariate" or callable')
    return sse


def find_changepoints(x, y, n_changepoints, model='linear'):
    """
    find the changepoints for a model with n_changepoints fit to data (x,y)
    x,y are data arrays which should be the same length
    n_changepoints should be an integer < len(x)
    model should either be 'linear', 'constant' or 'multivariate' (for multivariate y)
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


def score_model_bayes(y, posterior_mu, posterior_sigma):
    """
    return the log probability of the posterior model N(mu, sigma) on the data y
    """
    return np.sum(scipy.stats.multivariate_normal.logpdf(y, posterior_mu, posterior_sigma))

def update_model(y, prior_mu, prior_sigma):
    """
    compute the posterior normal dist for the mean of period with data y
    posterior normal has mu and sigma, it's not the std of the data
    """
    if y.ndim > 2: raise ValueError('Y has too many axes. shape: {}'.format(y.shape))
    elif y.ndim == 1:
        y = y.reshape([-1,1])
        prior_mu = np.array([prior_mu]).reshape([1])
        prior_sigma = np.array([[prior_sigma]]).reshape([1,1])
        
    if len(y) < 2: return prior_mu, prior_sigma
    
    period_std = np.std(y, axis=0)
    
    posterior_mu = np.zeros_like(prior_mu)
    posterior_sigma = np.ones_like(prior_sigma)
    for i in range(len(prior_mu)):
        prior_inv_sigma_squared = 1 / (prior_sigma[i,i] ** 2)

        posterior_inv_sigma_squared = prior_inv_sigma_squared + len(y) / (period_std[i] ** 2)
        posterior_mu[i] = ((prior_inv_sigma_squared * prior_mu[i]) + (len(y) / period_std[i]**2) * np.mean(y[:,i])) / posterior_inv_sigma_squared
        posterior_sigma[i,i] = 1 / np.sqrt(posterior_inv_sigma_squared)
    return posterior_mu, posterior_sigma

def update_and_score(y, prior_mu, prior_sigma):
    #posterior_mu, posterior_sigma = update_model(y, prior_mu, prior_sigma)
    
    #return score_model_bayes(y, posterior_mu, posterior_sigma)
    
    # use evidence instead
    if y.ndim > 2: raise ValueError('Y has too many axes. shape: {}'.format(y.shape))
    
    
    elif y.ndim == 1:
        y = y.reshape([-1,1])
        prior_mu = np.array([prior_mu]).reshape([1])
        prior_sigma = np.array([[prior_sigma]]).reshape([1,1])
        
    if len(y) < 2: return score_model_bayes(y, prior_mu, prior_sigma)
    
    period_std = np.std(y, axis=0)
    data_inv_var = 1 / period_std**2
    
    log_score = 0
    for i in range(len(prior_mu)):
        prior_inv_sigma_squared = 1 / (prior_sigma[i,i] ** 2)
        posterior_inv_sigma_squared = prior_inv_sigma_squared + len(y) * data_inv_var[i]
        posterior_mu = ((prior_inv_sigma_squared * prior_mu[i]) + (len(y) / period_std[i]**2) * np.mean(y[:,i])) / posterior_inv_sigma_squared
        
        # from https://statproofbook.github.io/P/ugkv-lme
        log_score += (
              (len(y) / 2) * np.log(data_inv_var[i] / (2 * np.pi))
            + 0.5 * np.log(prior_inv_sigma_squared / posterior_inv_sigma_squared)
            - 0.5 * data_inv_var[i] * np.sum(y[:,i] ** 2)
            - 0.5 * prior_inv_sigma_squared * prior_mu[i]**2
            + 0.5 * posterior_inv_sigma_squared * posterior_mu**2
        )
    
    return log_score
        
    

def find_changepoint_posterior(x, y, n_changepoints, prior_over_periods=None,
                               prior_mu=0, prior_sigma=1):
    """
    find the changepoints for a model with n_changepoints fit to data (x,y)
    x,y are data arrays which should be the same length
    n_changepoints should be an integer < len(x)
    prior_over_periods should be an array of values between 0 and 1 indicating
    how certain we are that a changepoint will occur at each point in the range of x
    prior_mu, prior_sigma are the mean and std for the period means (NOT the data)
    period_std is the assumed standard deviation of the data within each period
    """
    # this is also a dynamic programming problem, except now it's all probabilities
    # Using Bayes rule, with n changepoints between i and j,  P(split right before k | data i...j) =
    # P(split right before k) * P(data i...k | MAP periodization on i...k-1 with n-1 changepoints) * P(data k...j | model fit to data k...j)
    
    # posterior(i,j,n) = max_k( prior(k) * likelihood(i,k) * posterior(k,j,n) )
    # likelihood(i,j) = update_and_score(data i...j, priors), store in posterior(i,j,0)
    
    # but everything's a sum because we do it in log-space
    x = np.array(x)
    y = np.array(y)
    
    if y.ndim > 1 and prior_mu.ndim == 1:
        prior_mu = np.ones(y.shape[-1]) * prior_mu
        prior_sigma = np.eye(y.shape[-1]) * prior_sigma
    
    
    assert len(x) == len(y)
    if n_changepoints == 0:
        return [], update_and_score(y, prior_mu, prior_sigma), []

    candidates = sorted(set(x))
    
    if prior_over_periods is None:
        prior_over_periods = np.ones(len(candidates)) / len(candidates)
    else:
        if len(prior_over_periods) != len(candidates):
            raise ValueError(
                'Priors given for {} splits for data with {} X values'.format(
                len(prior_over_periods), len(candidates))
            )
    
    posterior_table = np.zeros([len(candidates), len(candidates)+1, n_changepoints]) - np.inf
    choices = np.zeros([len(candidates), len(candidates)+1, n_changepoints],dtype=int)
    
    # base cases
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)+1):
            x_masked, y_masked = range_filter(x,y,candidates[i],candidates[min(j, len(candidates)-1)])
            if len(x_masked) == 0: continue
            else:
                likelihood = update_and_score(y_masked, prior_mu, prior_sigma)
                if np.isnan(likelihood):
                    posterior_table[i,j,0] = -np.inf
                else:
                    posterior_table[i,j,0] = likelihood
             
                    
    # recursive cases
    for n in range(1, n_changepoints):
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)+1):
                options = np.arange(i+1,j)
                if len(options) == 0:
                    posterior_table[i,j,n] = -np.inf
                else:
                    posteriors = np.log(prior_over_periods[options]/np.sum(prior_over_periods[options])) + posterior_table[i,options,0] + posterior_table[options,j,n-1]
                    choice = options[np.argmax(posteriors)]
                    posterior_table[i,j,n] = np.max(posteriors)
                    choices[i,j,n] = choice
    
    # recover solution
    options = np.arange(len(candidates))
        
    optimal_split = np.argmax(posterior_table[0,options,0] + posterior_table[options,len(candidates),n_changepoints-1] + np.log(prior_over_periods[options]/np.sum(prior_over_periods[options])))
    log_map = posterior_table[0,optimal_split,0] + posterior_table[optimal_split,len(candidates),n_changepoints-1] + np.log(prior_over_periods[options]/np.sum(prior_over_periods[options]))
    splits = [optimal_split]
    for n in range(n_changepoints-1,0,-1):
        splits.append(choices[splits[-1],len(candidates),n])
    return [candidates[s] for s in splits], log_map, posterior_table


def model_likelihood(x, y, splits, prior_mu, prior_sigma):
    """
    compute the likelihood of the given splits under the N(prior_mu, prior_sigma^2) data model
    """
    assert len(x) == len(y)
    
    changepoints = [np.nanmin(x)] + list(splits) + [np.nanmax(x)]
    
    total_log_likelihood = 0
    
    for cp in range(len(changepoints)-1):
        x_masked, y_masked = range_filter(x,y,changepoints[cp],changepoints[cp+1])
        
        total_log_likelihood += update_and_score(y_masked, prior_mu, prior_sigma)
    return total_log_likelihood