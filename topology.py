import numpy as np
import torch


def sample_W(W, nSamples, isRandom=True):
    n = W.shape[0]
    random_indices = np.random.choice(n, size=nSamples, replace=False)
    return W[random_indices]

def calculate_ph_dim(W, min_points=200, max_points=1000, point_jump=50,  
        h_dim=0, print_error=False):
    from ripser import ripser
    # sample_fn should output a [num_points, dim] array
    
    # sample our points
    test_n = range(min_points, max_points, point_jump)
    lengths = []
    for n in test_n:
        diagrams = ripser(sample_W(W, n))['dgms']
        
        if len(diagrams) > h_dim:
            d = diagrams[h_dim]
            d = d[d[:, 1] < np.inf]
            lengths.append((d[:, 1] - d[:, 0]).sum())
        else:
            lengths.append(0.0)
    lengths = np.array(lengths)
    
    # compute our ph dim by running a linear least squares
    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = y.mean() - m * x.mean()
    
    error = ((y - (m * x + b)) ** 2).mean()
    
    if print_error:
        print(f"Ph Dimension Calculation has an approximate error of: {error}.")
    return 1 / (1 - m)

def calculate_ph_dim_gpu(W, min_points=200, max_points=1000, 
        point_jump=50, h_dim=0, print_error=False):
    from torchph.pershom import vr_persistence
    # sample_fn should output a [num_points, dim] array
    
    # sample our points
    test_n = range(min_points, max_points, point_jump)
    lengths = []
    for n in test_n:
        samples = sample_W(W, n)
        dist_matrix = torch.cdist(samples, samples)
        
        d, _ = vr_persistence(dist_matrix, 0, 0)
        d = d[0]
        lengths.append((d[:, 1] - d[:, 0]).sum())

    lengths = torch.stack(lengths)
    
    # compute our ph dim by running a linear least squares
    x = torch.tensor(test_n).to(lengths).log()
    y = lengths.log()
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = y.mean() - m * x.mean()
    
    error = ((y - (m * x + b)) ** 2).mean()
    
    if print_error:
        print(f"Ph Dimension Calculation has an approximate error of: {error}.")
    return 1 / (1 - m)
