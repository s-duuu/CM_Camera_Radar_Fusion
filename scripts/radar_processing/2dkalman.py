from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd
import csv
import os

os.chdir('/home/heven/CoDeep_ws/src/yolov5_ros/src/csv')

distance_data = []
velocity_data = []
f = open("/home/heven/CoDeep_ws/src/yolov5_ros/src/csv/fusion_distance.csv")

rea = csv.reader(f)
for row in rea:
    distance_data.append(row)
f.close

f2 = open("/home/heven/CoDeep_ws/src/yolov5_ros/src/csv/veclocity.csv")

rea = csv.reader(f2)
for row in rea:
    velocity_data.append(row)
f2.close

# print(float(distance_data[0][0]))
# print(type(float(distance_data[0][0])))

kf = KalmanFilter(dim_x=2, dim_z=2)

kf.x = np.array([[float(distance_data[0][0])],
                [float(velocity_data[0][0])]])

kf.F = np.array([[1, 1/30],
                [0, 1]])

kf.P = np.array([[25., 0.], [0., 0.000000000001]])

kf.Q = np.array([[500, 0.], [0., 500]])

kf.H = np.array([[1., 0.], [0., 1.]])

kf.R = np.array([[25., 0.], [0., 5000000]])

filtered_distance = []
filtered_velocity = []

for i in range(679):
    z = np.array([[float(distance_data[i][0])], [float(velocity_data[i][0])]])
    
    kf.predict(u=None, B=None, F=kf.F, Q=kf.Q)
    kf.update(z, kf.R, kf.H)

    filtered_distance.append(kf.x[0][0])
    filtered_velocity.append(kf.x[1][0])


df = pd.DataFrame({"Distance" : filtered_distance, "Velocity" : filtered_velocity})

df.to_csv("kalman_result.csv", index=False)



