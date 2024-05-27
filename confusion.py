# functions to calculate dissimilarity measures
import numpy as np
def Ptilde(sector_temperatures, x_index, logprob):
    '''
    sector_temperatures: corresponds to a list of temperatures
    x_index: corresponds to a temperature and sample index (temperature, index)
    logprob: the list of logprobs caluclated above
    '''
    k, kkk = x_index
    ptilde = 0.
    for kk in sector_temperatures: ptilde += np.exp(logprob[(k, kk, kkk)])
    
    return ptilde    


### general g-dissimilarity indicator

def general_indicator(jT, L, Ts, ys, res, g = lambda p: p):
    '''
    jT: index of temperature of split. convention: leftmost point
    L: number of points per sector.
    Ts: parameterization of x-axis
    ys: text samples. only used to infer number of samples.
    res: pre calculated logarithmic probabilities.
    g: g-dissimilarity function from article.

    returns loss and variance_of_sample_mean for each T. for final result, this needs to be averaged.
    '''
    key = list(ys.keys())[0]; n_iter = len(ys[key]) # infer number of samples n_iter
    
    sector = {} # list of left and right temperatures
    sector['left'] = [Ts[jT+l] for l in range(L)] 
    sector['right'] = [Ts[jT+l+L] for l in range(L)]
    
    losses_mean = {}
    losses_var = {}
    for T in sector['left']+sector['right']:
        losses = []
        for i in range(n_iter):
            ptl = Ptilde(sector['left'], (T, i), res)
            ptr = Ptilde(sector['right'], (T, i), res)

            pl = ptl / (ptl+ptr); pr = ptr / (ptl+ptr)
            if T in sector['right']: loss = g(pr)
            else: loss = g(pl)
            
            if (ptl+ptr)>0: losses.append(loss)

        losses_mean[T] = np.mean(losses)
        losses_var[T] = np.var(losses, ddof = 1)/len(losses)
    return losses_mean, losses_var

def stat_indicator_scan(Ts,ys,res,L=1,jmax=1000, g = lambda p: p):
    '''
    returns scanned losses and sample mean variances for each T.
    for final result, this needs to be averaged over sector.
    '''
    all_confs = []
    all_errs = []
    for j in range(jmax):
        try:
            c, e = general_indicator(j,L,Ts,ys,res,g)
            
            losses = np.array([prob for prob in c.values()])
            errors = np.array([err for err in e.values()])
            
            all_confs.append(losses)
            all_errs.append(errors)
            
        except:
            print(f'{j} failed')
    return all_confs, all_errs


def stat_indicator_scan_final(Ts,ys,res,L=1,jmax=1000, g = lambda p: p):
    '''
    the final result we want to plot, i.e. loss(T) and also standard_error(T)
    '''
    confs, vs = stat_indicator_scan(Ts,ys,res,L,jmax,g)
    return np.array([np.mean(scan[:]) for scan in confs]), np.sqrt(np.array([np.mean(scan[:])/len(scan[:]) for scan in vs]))

