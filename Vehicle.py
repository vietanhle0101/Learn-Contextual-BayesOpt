import numpy as np
from util import *

class Vehicle:

    def __init__(self, p0, v0, bounds, T):
        self.p = p0; self.v = v0
        self.state = np.array([p0, v0])
        self.u = 0.0
        self.T = T; self.Ts = 0.02
        self.v_min = bounds['v_min']; self.v_max = bounds['v_max']
        self.u_min = bounds['u_min']; self.u_max = bounds['u_max']        
        self.X_hist = np.empty([2,0], dtype=np.ndarray)
        self.U_hist = np.empty([0], dtype=float)
        self.T_hist = np.empty([0], dtype=float)
        self.X_hist = np.hstack([self.X_hist, self.state.reshape([-1,1])])

    def run(self, u):
        u = max(self.u_min, min(self.u_max, u))
        v = self.state[1]
        if v + self.T*u > self.v_max:
            u = (self.v_max-v)/self.T
        elif v + self.T*u < self.v_min:
            u = (self.v_min-v)/self.T
        
        self.p += self.v*self.Ts + 0.5*u*self.Ts**2 
        self.v += u*self.Ts
        self.state[0] = self.p; self.state[1] = self.v 
        self.u = u

    def save_data(self, time_stamp):
        self.X_hist = np.hstack([self.X_hist, self.state.reshape([-1,1])])
        self.U_hist = np.hstack([self.U_hist, self.u])  
        self.T_hist = np.hstack([self.T_hist, time_stamp])  

    def print(self, i):
        print("Some information for vehicle", i, ":", self.p, self.v)
    
    def const_U(self, U, t):
        time = np.arange(t, t+self.T, self.Ts)
        for _ in time:
            self.run(U)  
        self.save_data(t+self.T)
        
    def const_V(self, V, t):
        time = np.arange(t, t+self.T, self.Ts)
        for k in time:
            U = (V-self.v)/self.Ts
            self.run(U, k)  
        self.save_data(t+self.T)   

class CAV(Vehicle):
    
    def __init__(self, p0, v0, bounds, T):
        super().__init__(p0, v0, bounds, T)
        self.type = "CAV" # of course

class HDV(Vehicle):
    
    def __init__(self, p0, v0, bounds, T):
        super().__init__(p0, v0, bounds, T)
        self.type = "HDV" # of course

    def IRL(self, CAV_state, Q, v_ref = None):
        p = self.p; v = self.v
        if v_ref is None:
            u = IRL_CFM(p, v, CAV_state[0], CAV_state[1], 0.0, Q, self.T, self.v_max, 0.0)
        else:
            u = IRL_CFM(p, v, CAV_state[0], CAV_state[1], 0.0, Q, self.T, v_ref, 0.0)

        u = max(self.u_min, min(self.u_max, u))
        if v + self.T*u < self.v_min:
            u = (self.v_min-v)/self.T
#         elif v + self.T*u > self.v_max:
#             u = (self.v_max-v)/self.T
        return u
    
## This function is to compute the action of the human driver by IRL model
def IRL_CFM(p2_0, v2_0, p1_0, v1_0, u1_0, W, T = .1, v_ref = 0.5, r = 0.):
    # Note that p2_0 and p1_0 should be the distance to the conflict point
    C = np.zeros(7)
    C[0] = W[0] + W[1]*T**2
    C[1] = 2*W[1]*(v2_0-v_ref)*T
    C[2] = W[1]*(v2_0-v_ref)**2
    C[3] = W[2]
    C[4] = (0.5*T**2)**2
    C[5] = 2*0.5*T**2*(p2_0+T*v2_0)
    C[6] = ((p1_0+v1_0*T)**2+(p2_0+v2_0*T)**2-r**2)  
    
    coeff = [(2*C[0]*C[4]), (C[1]*C[4]+2*C[0]*C[4]), (C[1]*C[5]+2*C[0]*C[6]-2*C[3]*C[4]), \
                (C[1]*C[6]-C[3]*C[5])]
    root = np.roots(coeff)
    sol = np.real(root[np.isreal(root)])[0]
    return sol        

## Constant time headway car-following model
def CTH(dist, d_min, rho, v_min = 0.0, v_max = 0.5):
    return min(max((dist-d_min)/rho, v_min), v_max)