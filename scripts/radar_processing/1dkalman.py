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

kf = KalmanFilter(dim_x=2, dim_z=1)

kf.x = np.array([[0],
                [float(velocity_data[0][0])]])

kf.F = np.array([[1, 1/30],
                [0, 1]])

# kf.H = np.array([[1.0, 0.]])

kf.H = np.array([[0., 1.]])

kf.R = 100000

kf.P = 0.0000000001

kf.Q = 500

filtered_distance = []
filtered_velocity = []

prev_v = float(velocity_data[0][0])

# for distance in distance_data:
#     z = np.array([[float(distance[0])]])
    
#     kf.predict(u=None, B=None, F=kf.F, Q=kf.Q)
#     kf.update(z, kf.R, kf.H)

#     filtered_distance.append(kf.x[0][0])

for velocity in velocity_data:
    kf.x = np.array([[0], [prev_v]])
    z = np.array([[float(velocity[0])]])
    
    kf.predict(u=None, B=None, F=kf.F, Q=kf.Q)
    kf.update(z, kf.R, kf.H)

    filtered_velocity.append(kf.x[1][0])

    prev_v = kf.x[1][0]


# df = pd.DataFrame({"Velocity" : filtered_distance})
# df.to_csv("1d_kalman_distance_result.csv", index=False)

df2 = pd.DataFrame({"Velocity" : filtered_velocity})
df2.to_csv("1d_kalman_velocity_test.csv", index=False)
