import numpy as np
import casadi as ca
from copy import deepcopy
import numpy.linalg as LA

class MPC:
    p_offset = -0.2
    
    def __init__(self, H, T, bounds):
        self.H = H; self.T = T
        self.u_min = bounds['u_min']; self.u_max = bounds['u_max']
        self.v_min = bounds['v_min']; self.v_max = bounds['v_max']

    def set_state(self, all_st):
        self.st = all_st
        
    def set_input(self, all_in):
        self.input = all_in
        
    def set_ref(self, v_ref):
        self.v_ref = v_ref
        
    def set_safe_thres(self, r, rho = 0.2):
        self.r = r
        self.rho = rho
        
    def set_weight(self, weight):
        self.weight = weight
        
    def formulateMPC(self, n_HDV = 1):
        # Create MPC optimizer
        self.mpc_opti = ca.Opti()
        n_cars = n_HDV+1
        
        # Variables for CAV & HDV
        self.U = self.mpc_opti.variable(n_cars, self.H)
        self.V = self.mpc_opti.variable(n_cars, self.H+1)
        self.P = self.mpc_opti.variable(n_cars, self.H+1)
            
        self.V0 = self.mpc_opti.parameter(n_cars) # initial condition of V
        self.P0 = self.mpc_opti.parameter(n_cars) # initial condition of P
        self.VR = self.mpc_opti.parameter(n_cars) # V_ref
        
        # Objective function
        self.J = 0
        for k in range(self.H+1):
            for n in range(n_cars):
                if k != self.H:
                    self.J += self.weight['W1'][n]*self.U[n,k]**2 
                self.J += self.weight['W2'][n]*(self.V[n,k] - self.VR[n])**2
                if n != 0:
                    self.J += -self.weight['W12']*ca.log((self.P[0,k]-self.p_offset)**2 + (self.P[n,k]-self.p_offset)**2 + 1e-6)
        
        self.mpc_opti.minimize(self.J) 
        
        # Dynamics
        for i in range(n_cars):
            self.mpc_opti.subject_to(self.P[i,0] == self.P0[i])
            self.mpc_opti.subject_to(self.V[i,0] == self.V0[i])
            for k in range(self.H):
                self.mpc_opti.subject_to(self.P[i,k+1] == self.P[i,k] + self.T*self.V[i,k] \
                                         + 0.5*self.T**2*self.U[i,k])
                self.mpc_opti.subject_to(self.V[i,k+1] == self.V[i,k] + self.T*self.U[i,k])

         # Constraints
        self.mpc_opti.subject_to(self.v_min <= self.V[0,:])
        self.mpc_opti.subject_to(self.V[0,:] <= self.v_max)
        self.mpc_opti.subject_to(self.u_min <= ca.reshape(self.U, n_cars*self.H, 1))
        self.mpc_opti.subject_to(ca.reshape(self.U, n_cars*self.H, 1) <= self.u_max)
        # self.mpc_opti.subject_to(self.r**2 <= self.P[0,:]**2 + self.P[1,:]**2 - self.rho*self.V[0,:]**2)

        p_opts = {'verbose_init': False, 'print_time': 0}
        s_opts = {'tol': 0.001, 'print_level': 0, 'max_iter': 1000}
        self.mpc_opti.solver('ipopt', p_opts, s_opts)

        # Warm up
        self.mpc_opti.set_value(self.P0, self.st[0,:])
        self.mpc_opti.set_value(self.V0, self.st[1,:])
        self.mpc_opti.set_value(self.VR, self.v_max*np.ones(n_cars))
        
        try:
            sol = self.mpc_opti.solve()
        except RuntimeError:
            # print("An exception occurred")
            self.mpc_opti.set_initial(self.U, self.mpc_opti.debug.value(self.U))
        else:
            self.mpc_opti.set_initial(self.U, sol.value(self.U))
            self.mpc_opti.set_initial(self.V, sol.value(self.V))
            self.mpc_opti.set_initial(self.P, sol.value(self.P))
            
        # print(sol.value(self.U))
        
    def solveMPC(self):
        self.mpc_opti.set_value(self.P0, self.st[0,:])
        self.mpc_opti.set_value(self.V0, self.st[1,:])

        try:
            sol = self.mpc_opti.solve()
        except RuntimeError:
            # print("An exception occurred")
            self.u = self.mpc_opti.debug.value(self.U)
        else:
            self.u = sol.value(self.U)
            self.mpc_opti.set_initial(self.U, np.vstack((sol.value(self.U)[1:,:], sol.value(self.U)[-1:,:])))    
            self.mpc_opti.set_initial(self.V, np.vstack((sol.value(self.V)[1:,:], sol.value(self.V)[-1:,:])))    
            self.mpc_opti.set_initial(self.P, np.vstack((sol.value(self.P)[1:,:], sol.value(self.P)[-1:,:])))    
                      
        return self.u[0][0]   



