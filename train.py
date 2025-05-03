"""
This script is to train the policy with nested-loop contextual BO
"""


import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from datetime import datetime

from util import *
from Vehicle import *
from MPC import *
from BayesOpt_torch import *
from examples.ex_fcn import blackbox_fcn

import warnings
import logging
logging.disable(logging.CRITICAL)
warnings.filterwarnings('ignore')
import argparse

def single_sim(init_cond, W_A, W_H, W_AH, W_metric, T = 0.2, H = 20, L = 200):
    # np.random.seed(28)
    bounds = {'v_min' : 0.0, 'v_max' : 0.5, 'u_min' : -1.0, 'u_max' : 0.8}
    CZ = {'start': -2.0, 'end': 1.0}
    r = 0.2; rho = 0.2

    init_pos = init_cond[:2]; init_vel = init_cond[2:]
    Cars = []
    Cars.append(CAV(init_pos[0], init_vel[0], bounds, T))
    Cars.append(HDV(init_pos[1], init_vel[1], bounds, T))

    control = MPC(H, T, bounds)
    all_st = np.vstack([car.state for car in Cars]).T
    control.set_state(all_st)
    control.set_safe_thres(r, rho)
    W_irl = np.hstack([10**W_H, 10**W_AH])
    control_weight = {
        'W1': np.array([10**W_A[0], 10**W_H[0]]),
        'W2': np.array([10**W_A[1], 10**W_H[1]]),
        'W12': 10**W_AH
    }
    control.set_weight(control_weight)
    control.formulateMPC()

    # Cars[0].const_U(0.0, 0.0)
    # Cars[1].const_U(Cars[1].IRL(Cars[0].state, W_irl), 0.0)

    ## Main loop
    t_f0 =  T*L; t_f1 = T*L; dist = []
    energy = energy_model(Cars[0].u, Cars[0].v, T)

    for k in range(L):
        t = (k+1)*T
        # print("Time %s" %t)

        # Update measurement
        all_st = np.vstack([car.state for car in Cars]).T
        control.set_state(all_st)

        # Now let's run
        u1 = control.solveMPC()
        u2 = Cars[1].IRL(Cars[0].state, W_irl)

        Cars[0].const_U(u1, t)
        Cars[1].const_U(u2, t)

        dist.append(Cars[0].p**2 + Cars[1].p**2 - rho*Cars[0].v**2)
        energy += energy_model(Cars[0].u, Cars[0].v, T)

        # Find exit time for each vehcile
        if Cars[0].p > CZ['end']:
            t_f0 = t - (Cars[0].p - CZ['end'])/Cars[0].v
            break   
            
    # print(t_f0, energy, np.min(np.array(dist) - r))
    true_cost = W_metric[0]*t_f0 + W_metric[1]*energy + W_metric[2]*soft_max(-np.min(np.array(dist) - r**2)) 
    # Can use either: soft_max(-np.min(np.array(dist) - r)) OR sigmoid(-np.min(np.array(dist) - r)) 
    return true_cost

def multi_sim(samples, W_A, W_H, W_AH, W_metric):
    all_cost = []

    for idx, sample in enumerate(samples):
        init_cond = sample
        true_cost = single_sim(init_cond, W_A, W_H, W_AH, W_metric)
        all_cost.append(true_cost)

    average_cost = np.mean(all_cost)
    print("Average cost is ", average_cost)
    return average_cost

class MPC_true_cost(blackbox_fcn):
    r"""black-box function for BayesOpt.
    """
    def __init__(self, gridsize = [3,3], W_metric = [1, 5e1, 1e3]):
        super().__init__()
        CZ = {'start': -2.0, 'end': 1.0}
        pp = CZ['start'] + np.linspace(0.0, 0.5, gridsize[0])
        vv = np.linspace(0.2, 0.5, gridsize[1])
        grid = np.meshgrid(pp, pp, vv, vv)
        self.samples = [[grid[0].ravel()[i], grid[1].ravel()[i], grid[2].ravel()[i], grid[3].ravel()[i]] for i in range(gridsize[0]**2*gridsize[1]**2)]        
        self.W_metric = W_metric

    def evaluate_true(self, Z, T):
        W_AH = 0.0
        obj = multi_sim(self.samples, Z, T, W_AH, self.W_metric) # W_A = Z; W_H = T; 
        return - obj 
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_max", type=int, default=35)
    parser.add_argument("--k_init", type=int, default=5)
    parser.add_argument("--j_max", type=int, default=35)
    parser.add_argument("--j_init", type=int, default=5)
    parser.add_argument("--beta", type=float, default=1e2)
    parser.add_argument("--beta_rate", type=float, default=0.9)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--kernel_OM", type=str, default="RBF") # May use "RBF" or "Matern32"
    parser.add_argument("--kernel_SM", type=str, default="DLK") # May use "DLK", "RBF" or "Matern32"
    parser.add_argument("--wt", type=float, default=1e0)
    parser.add_argument("--we", type=float, default=3e0)
    parser.add_argument("--wc", type=float, default=1e4)
    parser.add_argument("--grid_level", type=int, default=3)

    input_parser = parser.parse_args()
    k_max = input_parser.k_max; k_init = input_parser.k_init
    j_max = input_parser.j_max; j_init = input_parser.j_init
    beta = input_parser.beta; beta_rate = input_parser.beta_rate
    verbose = input_parser.verbose
    kernel_OM = input_parser.kernel_OM
    kernel_SM = input_parser.kernel_SM
    wt = input_parser.wt; we = input_parser.we; wc = input_parser.wc
    grid_level = input_parser.grid_level

    n_pts = 101
    # Bounds on the inputs variable
    var_bounds = [(-2., 2.), (-2., 2.)]
    # Bounds on the context variable
    ctx_bounds = [(-2., 2.), (-2., 2.)]
    n_var = 2; n_ctx = 2
    f1 = MPC_true_cost(gridsize = [grid_level, grid_level], W_metric = [wt, we, wc])
    training_params = {"max_iter": 200, "optimizer": "Adam", "num_restarts": 5}
    CBO = Context_BayesOpt(n_var, n_ctx, print_level = verbose, 
            kernel_OM = kernel_OM, kernel_SM = kernel_SM, training_params=training_params)
    CBO.set_feasible_set(var_bounds, ctx_bounds, n_pts)

    OUTCOME = CBO.run_Outer(f1, k_max=k_max, k_init=k_init, j_max=j_max, j_init=j_init, beta=beta, beta_rate=beta_rate, tol=1e-1, patience=3)

    # Returns the current local date
    now = datetime.now() # current date and time
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    print("Today date is: ", date_time)
    filename = "output/" + date_time + "_" + str(wt) + "_" + str(we) + "_" + str(wc) + ".csv"
    np.savetxt(filename, np.hstack(OUTCOME))

if __name__ == "__main__":
    main()
