import matplotlib.pyplot as plt
import pickle
import numpy as np

x = pickle.load(open("data/point","rb"))

print(np.array(x["av"]).shape)

plt.plot(np.array(x["av"]))
plt.ylim(-1e-2,1e-2)
plt.show()
