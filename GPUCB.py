import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# import time

class gp_ucb():
    
    def __init__(self,x1_values,x2_values,observations,kernel=('rbf',2),sw=None):
        self.x1_values = x1_values
        self.x2_values = x2_values
        self.sw = sw
        self.observations = observations
        self.configurations = len(x1_values) * len(x2_values)
        self.config_stats = dict()
        self.action_distance = dict()
        cnt_i = 0
        cnt = 0
        for i in x1_values:
            cnt_j = 0
            for j in x2_values:
                self.action_distance[(i,j)] = [np.array([cnt_i,cnt_j]), cnt]
                self.config_stats[(i,j)] = []
                cnt_j += 1
                cnt += 1
            cnt_i += 1
        self.kernels = self.kernel_function(self.action_distance,kernel,self.configurations)
        self.x1 = np.random.choice(x1_values)
        self.x2 = np.random.choice(x2_values)
        self.selected_actions = [self.action_distance[(self.x1,self.x2)][1]]
        self.action_sequence = []
        self.K_T = [[self.kernels[self.action_distance[(self.x1,self.x2)][1],self.action_distance[(self.x1,self.x2)][1]]]]
        self.y_t = []
        self.cov_x = []
        self.mean_x = []
        self.reg = []
        
    
    def __repr__(self):
        action_freq = dict()
        for action in self.action_sequence:
            if action not in action_freq:
                action_freq[action] = 1
            else:
                action_freq[action] += 1    
        
        out = 'f_min: '+str(self.f_min)+'\n' + \
                'x_min: '+str(self.x_min)+'\n' + \
                'violation: '+str(self.violation)+'\n' + \
                'time: '+str(self.time)
        return out
    
        
    
    def kernel_function(self,action_distance,kernel,configurations):
        kernels = np.zeros((configurations,configurations))
        for key1, value1 in action_distance.items():
            for key2, value2 in action_distance.items():
                if kernel[0] == 'rbf':
                    rho = kernel[1]
                    kernels[value1[1],value2[1]] = np.exp(-np.linalg.norm(value1[0]-value2[0])**2 / (2*rho**2))
        return kernels

    def k_time(self,x):
        k_T = []
        for i in range(0,len(self.selected_actions)):
            k_T.append(self.kernels[self.selected_actions[i],x])
        return np.array(k_T)

    def mu_T(self,x,sigma):
        k_T = self.k_time(x)
        #print(k_T)
        temp = np.linalg.inv(self.K_T + sigma*np.identity(len(k_T)))
        #print(temp)
        mu_T = np.dot(np.dot(k_T,temp),np.array(self.y_t))
        return mu_T

    def cov_T(self,x1,x2,sigma):
        k1_T = self.k_time(x1)
        k2_T = self.k_time(x2)
        temp = np.linalg.inv(self.K_T+sigma*np.identity(len(k1_T)))
        k_T = self.kernels[x1,x2] - np.dot(np.dot(k1_T,temp),k2_T)
        return max(0,k_T)

    def update_K_T(self,x):
        vec1 = [self.kernels[x,i] for i in self.selected_actions]
        self.K_T.append(vec1)
        cnt = 0
        for l in self.K_T:
            x_l = self.selected_actions[cnt]
            l.append(self.kernels[x_l,x])
        return self

    def remove_K_T(self):
        self.K_T = self.K_T[-self.sw:]
        for i in range(0,len(self.K_T)):
            self.K_T[i] = self.K_T[i][-self.sw:]
        return self
        
    
    def calculate_action(self,sigma,t):
        max_f = -1000   # function evaluation
        max_x = -1      # action index
        x1 = self.x1_values[0]
        x2 = self.x2_values[0]
        beta = np.log((32*np.pi*t) / 6)
        # keep only measurements of the last sw slots
        if self.sw != None:
            if len(self.y_T) > self.sw:
                self.y_T = self.y_T[-self.sw:]
                self.actions_selected = self.actions_selected[-self.sw:]
                self.K_T = self.remove_K_T(self.K_T,self.sw)
        for key, value in self.action_distance.items():
            x = value[1]
            mean_x = self.mu_T(x,sigma)
            cov_x = self.cov_T(x,x,sigma)
            f = mean_x + np.sqrt(beta*cov_x)
            if f > max_f:
                max_f = f
                max_x = x
                x1,x2 = key[0],key[1]         
        self.x1 = x1
        self.x2 = x2
        self.action_sequence.append((x1,x2))
        return max_x
    
    
    def iterate(self,sigma,iters):
        for t in tqdm(range(0,iters),position=0):
            reward = np.random.choice(self.observations[(self.x1,self.x2)])
            self.y_t.append(reward)
            x = self.calculate_action(sigma,t+1)
            self.update_K_T(x)
            self.selected_actions.append(x)
            
    
    def calculate_regret(self,fun_evals):
        max_reward = np.max(fun_evals)
        total_reg = 0
        for action in self.action_sequence:
            i,j = action[0],action[1]
            reward = fun_evals[j,i]
            total_reg += (max_reward-reward)
            self.reg.append(total_reg)
            
        fig = plt.figure()
        plt.plot(range(0,len(self.action_sequence)), self.reg, 'k-', lw=2)
        plt.grid(True)
        plt.ylabel('Total Regret' , fontsize=20)
        plt.xlabel('Slot', fontsize=20)
        plt.tight_layout()
        plt.show()
        #fig.savefig('regret.eps', format='eps')
        
        
        
        
        
            

        

