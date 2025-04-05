import sklearn
import random
import numpy as np
import os

beta = np.zeros(50) 
for i in range(5):
    index = random.choice(range(50))
    beta[index] = random.random()

print(beta)