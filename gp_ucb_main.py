import numpy as np
import matplotlib.pyplot as plt
from GPUCB import gp_ucb
import seaborn as sns
sns.set()


# Range of x and y values
x1_values = list(range(0,4,1))
x2_values = list(range(0,8,1))


# Average values of maximization function at (x,y) points
fun_evals = np.array([[-9.28443086, -7.61440994, -6.04522563, -1.2437448 ],
       [-6.79203785, -5.31388005, -3.79497582, -0.47589849],
       [-4.87107309, -3.62555952, -2.33082936, -0.02650078],
       [-2.44000715, -1.20364123, -0.2179125 , -0.37091677],
       [-0.10678793,  0.77568399,  1.27875152, -0.68863567],
       [ 1.46112736,  1.75769733,  1.76357373, -1.28447495],
       [ 1.31541354,  1.49205053,  1.30313236, -1.70240595],
       [-0.20974405, -0.18067325, -0.34514184, -3.07301368]])

# Create a collection of measurements at each point with Gaussian error
observations = dict()
sigma = 2
for i in range(0,len(x2_values)):
    for j in range(0,len(x1_values)):
        if (x1_values[j],x2_values[i]) not in observations:
            observations[(x1_values[j],x2_values[i])] = np.random.normal(fun_evals[i,j],sigma,1000)
        

iterations = 100
gp = gp_ucb(x1_values,x2_values,observations,kernel=('rbf',2),sw=None)
gp.iterate(sigma,iterations)
gp.calculate_regret(fun_evals)




fig1 = plt.figure()
ax = sns.heatmap(fun_evals, annot=True, annot_kws={"size": 8}, linewidth=0.5, cmap="YlGnBu")
plt.ylabel('$x_2$ values' , fontsize=20)
ax.set_yticklabels(x2_values, rotation=0, fontsize=16)
plt.ylim([0,8])
ax.set_xticklabels(x1_values, rotation=0, fontsize=16)
#plt.yticks(np.arange(len(N_values)+1),N_values)
plt.xlabel('$x_1$ values', fontsize=20)
#plt.xticks(range(0,len(Q_values)),Q_values)
plt.tight_layout()
plt.show()
#fig1.savefig('fun_average.eps', format='eps')
