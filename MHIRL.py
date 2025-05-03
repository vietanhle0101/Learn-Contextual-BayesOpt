import numpy as np
import numpy.linalg as LA

class MHIRL:
    def __init__(self, car, G = 20, epsi = 1e-6):
        self.car = car
        self.G = G; self.T = car.T  
        self.epsi = epsi

    def set_weights(self, WW):
        self.W = np.log10(WW)
            
    # Compute the objective
    def compute_objective(self, d1, d2, v1, v2, u2, WW, v_max):
        f_a = WW[0]*u2**2
        f_v = WW[1]*(v2-v_max)**2
        f_d = -WW[2]*np.log(d1**2 + d2**2 + 1e-6)
        return np.array([f_a, f_v, f_d])

    def IRL_CFM(self, p2_0, v2_0, p1_0, v1_0, u1_0, W, v_max = 0.5, r = 0.):
        # Note that p2_0 and p1_0 should be the distance to the conflict point
        C = np.zeros(7)
        C[0] = W[0] + W[1]*self.T**2
        C[1] = 2*W[1]*(v2_0-v_max)*self.T
        C[2] = W[1]*(v2_0-v_max)**2
        C[3] = W[2]
        C[4] = (0.5*self.T**2)**2
        C[5] = 2*0.5*self.T**2*(p2_0+self.T*v2_0)
        C[6] = ((p1_0+v1_0*self.T)**2+(p2_0+v2_0*self.T)**2-r**2+1e-6)  
        
        coeff = [(2*C[0]*C[4]), (C[1]*C[4]+2*C[0]*C[4]), (C[1]*C[5]+2*C[0]*C[6]-2*C[3]*C[4]), \
                (C[1]*C[6]-C[3]*C[5])]
        root = np.roots(coeff)
        sol = np.real(root[np.isreal(root)])[0]
        return sol

    def learn_weights(self, DATA, n_iter = 10, zeta = 1e2, decay = 0.01):
        v_max = self.car.v_max
        G = min(self.G, DATA["U"].shape[0])
        if G > 1:
            W_old = self.W
            for n in range(n_iter):
                zeta *= (1-decay)
                f_all = np.zeros([3,self.G])
                f_sol_all = np.zeros([3,self.G])
                WW = 10**W_old
                for l in range(G):
                    p1 = DATA["P"][0,l:l+2]; p2 = DATA["P"][1,l:l+2]
                    v1 = DATA["V"][0,l:l+2]; v2 = DATA["V"][1,l:l+2]
                    u1 = DATA["U"][0,l:l+1]; u2 = DATA["U"][1,l:l+1]
                    u2_sol = self.IRL_CFM(p2[0], v2[0], p1[0], v1[0], 0.0, WW, v_max)
                    v2_pred = v2[0] + self.T*u2_sol
                    p2_pred = p2[0] + self.T*v2[0] + 0.5*self.T**2*u2_sol
                    ff = self.compute_objective(p1[1], p2[1], v1[1], v2[1], u2[0], WW, v_max)
                    f_all[:,l] = ff
                    ff = self.compute_objective(p1[1], p2_pred, v1[1], v2_pred, u2_sol, WW, v_max)
                    f_sol_all[:,l] = ff
                grad_f = np.mean(f_sol_all, axis = 1) - np.mean(f_all, axis = 1) # print(grad_f)
                grad_f = grad_f*10**W_old # print(np.exp(W_old)) # print(grad_f)
                new_W = np.maximum(-3.0, np.minimum(3.0, W_old + zeta*grad_f)) 

                # print(new_W, W_old) # print("____________________")
                if LA.norm(new_W - W_old) < self.epsi:
                    W_old = new_W
                    # print("Converged at", n)
                    break
                else:
                    W_old = new_W
            
            # Normalization
            self.W = W_old - W_old[-1]
            self.W = np.maximum(-3.0, np.minimum(3.0, self.W)) 
