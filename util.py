import numpy as np

r"""Some misc. functions
"""

def energy_model(U, V, dt, real = False):
    energy = 0.0
    gr = 10

    if real:
        b0 = 0.1569; b1 = 2.450*1e-2
        b2 = -7.415*1e-4; b3 = 5.975*1e-5
        c0 = 0.07224; c1 = 9.681*1e-2; c2 = 1.075*1e-3
        v = V; a = max(0, U)
        for _ in range(gr):
            v += U*dt/gr
            energy += (b0 + b1*v + b2*v**2 + b3*v**3 + a*(c0 + c1*v + c2*v**2))*dt/gr
    else:
        energy += U**2*dt
    return energy

def sigmoid(x, miu = 10.0):
    return 1/(1 + np.exp(-miu*x))

def soft_max(x, miu = 10.0):
    return np.log(np.exp(miu*x)+1)/miu

# intersection between line(p1, p2) and line(p3, p4)
def intersect(p1, p2, p3, p4):
    if p1 <= 0 and p2 >= 0 and p3 <= 0 and p4 >= 0:
        return True
    return False


def linearly_spaced_combinations(bounds, num_samples):
    """
    Return 2-D array with all linearly spaced combinations with the bounds.

    Parameters
    ----------
    bounds: sequence of tuples
        The bounds for the variables, [(x1_min, x1_max), (x2_min, x2_max), ...]
    num_samples: integer or array_likem
        Number of samples to use for every dimension. Can be a constant if
        the same number should be used for all, or an array to fine-tune
        precision. Total number of data points is num_samples ** len(bounds).

    Returns
    -------
    combinations: 2-d array
        A 2-d arrray. If d = len(bounds) and l = prod(num_samples) then it
        is of size l x d, that is, every row contains one combination of
        inputs.
    """
    num_vars = len(bounds)
    num_samples = [num_samples] * num_vars

    if len(bounds) == 1:
        return np.linspace(bounds[0][0], bounds[0][1], num_samples[0])[:, None]

    # Create linearly spaced test inputs
    inputs = [np.linspace(b[0], b[1], n) for b, n in zip(bounds,
                                                         num_samples)]

    # Convert to 2-D array
    return np.array([x.ravel() for x in np.meshgrid(*inputs)]).T
