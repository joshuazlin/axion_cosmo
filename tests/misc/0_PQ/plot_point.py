import matplotlib.pyplot as plt
import pickle
import numpy as np

x = pickle.load(open("data/point","rb"))

print(np.array(x["point"]).shape)

plt.plot(np.array(x["point"]))
plt.show()
