import matplotlib.pyplot as plt
import os
import csv

c = []
r = []
f = []

os.chdir('/home/heven/CoDeep_ws/src/yolov5_ros/src/csv/result')

with open('distance_fusion_result.csv', 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        c.append(float(row[1]))
        r.append(float(row[2]))
        f.append(float(row[3]))

print(c)

# print(len(c))
# print(len(r))
# print(len(f))

x = list(range(0, 748))
fig = plt.figure(figsize=(30, 13))
plt.plot(x, c, x, r, x, f)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(['Camera', 'Radar', 'Fusion'], fontsize= 28)

# ax1 = fig.add_subplot(1, 1, 1)
# ax1.plot(x, c, color = 'red', label='Camera')
# ax1.tick_params(axis='y', labelcolor="red")

# ax2 = fig.add_subplot(1, 1, 1)
# ax2.plot(x, r, color = "green", label='Radar')
# ax2.tick_params(axis='y', labelcolor="green")
# # ax2.legend(loc="upper right")

# ax3 = fig.add_subplot(1, 1, 1)
# ax3.plot(x, f, color = "blue", label='Fusion')
# ax3.tick_params(axis='y', labelcolor='blue')
# # ax3.legend(loc="upper right")

plt.show()