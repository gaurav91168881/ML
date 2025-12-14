"""    # GRADIENT DESCENT     """

import numpy as np

def gradient_descent(x, y):    # 'x' -> Feature | 'y' -> Target
    m = 0  # Intial Value of 'm'
    c = 0  # Intial Value of 'c'
    learning_rate = 0.05  # Arbitary value for alpha
    n = len(x)  # no. of rows | sample size
    no_of_steps = 1000  # Iterations

    for step_no in range(no_of_steps):
        y_pred = m * x + c

        cost = (1/n) * sum([val ** 2 for val in (y - y_pred)])
        
        dm = (-2/n) * sum(x * (y - y_pred))
        dc = (-2/n) * sum(y - y_pred)

        m = m - learning_rate * dm  # move towards downward dxn to find minima
        c = c - learning_rate * dc  # move towards downward dnx to find minima

        # print(f"Step No. = {step_no + 1} | Slope = {m} | Intercept = {c} | Cost = {cost}")
        print(f"Step No. = {step_no + 1} | Slope = {round(m,2)} | Intercept = {round(c,2)} | Cost = {round(cost,2)}")

x = np.array([1, 2, 3, 4, 5])   # Feature
y = np.array([3, 13, 11, 7, 25]) # Target

print(gradient_descent(x,y))
        

import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.show()

plt.plot(x, 3.8 * x + 0.4, color = 'red')
plt.show()








