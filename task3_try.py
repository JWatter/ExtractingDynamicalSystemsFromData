#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from utils import *


# In[11]:


# prepare data
x0_f = "./data/nonlinear_vectorfield_data_x0.txt"
x1_f = "./data/nonlinear_vectorfield_data_x1.txt"
x0 = pd.read_csv(x0_f, header=None, delimiter=" ").values
x1 = pd.read_csv(x1_f, header=None, delimiter=" ").values

del_t = 0.1
vec = (x1 - x0) / del_t


# ## Part One & Two

# In[17]:


Linear = LinearApprox()
v_hat = Linear.linear_approx(x0, vec)
x1_hat = v_hat * del_t + x0
mse = np.square(x1 - x1_hat).mean()
print("Linear Mean Squared Error: ", mse)


Nonlinear = NonlinearApprox(1000, 1)
v_hat = Nonlinear.radial_approx(x0, vec)
x1_hat = v_hat * del_t + x0
mse = np.square(x1 - x1_hat).mean()
print("Nonlinear Mean Squared Error: ", mse)



# ## Part Three
# In[27]:


T = 1
x_hat_pred = [x0]
v_hat = Nonlinear.predict(x_hat_pred[-1])

for _ in np.arange(T, step=del_t): 
    x_hat = v_hat * del_t + x_hat_pred[-1]
    x_hat_pred.append(x_hat)

x_hat_pred = np.dstack(x_hat_pred)

plt.figure(figsize=(5, 5))
for i in range(x_hat_pred.shape[0]):
    plt.scatter(x_hat_pred[i, 0, :], x_hat_pred[i, 1, :])
plt.show()



import matplotlib.animation as animation

fig, ax = plt.subplots()

#line_1, = ax.plot(x_1, y_1, linewidth=0.5, color='k')

x_1 = x_hat_pred[0, 0, :]
y_1 = x_hat_pred[0, 1, :]

line_1, = ax.plot(x_1, y_1, linewidth=0.5, color='k')

def update(num, x, y, line):
    """
    update function for matplotlib animation
    Plots trajectory for inputs x and y with attributes given by line
    """
    line.set_data(x_1, y_1)
    #line.axes.axis([0, 18, 10, 25])
    return line,

# Start animation
animation.FuncAnimation(fig, update, x_hat_pred.shape[0],
                              fargs=[x_1, y_1, line_1],
                              interval=20, blit=True, repeat=False)

plt.show()
