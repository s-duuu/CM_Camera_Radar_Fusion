import matplotlib.pyplot as plt
from datetime import datetime

fig = plt.figure()
ax = plt.axes(ylim = (0, 9))
for i in range(10):
    plt.plot(datetime.now(), i)
    plt.axis([0, 20, 0, 20])

    plt.show()