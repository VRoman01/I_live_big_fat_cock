#!../venv/bin/python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(1)
x = np.arange(100)
y = 2*x + 3 + 2*rng.randn(100)

fig = plt.figure()
ax = fig.add_subplot()