import pandas as pd
import numpy as np
from scipy import stats
import glob, os

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

dt = 1.0/60
#Transition
F = np.array([[1, -dt], [0, 1]])
#Observation
H = np.array([1, 0]).reshape(1, 2)
Q = np.array([[0.001, 0.000], [0.000, 0.003]])
R = np.array([0.03]).reshape(1, 1)

# # Transition
# F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
# # Observation
# H = np.array([1, 0, 0]).reshape(1, 3)
# Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
# R = np.array([0.5]).reshape(1, 1)

def compute_kalman(measurements):
    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
    predictions = []
    for z in measurements:
        predictions.append(np.dot(H,  kf.predict())[0])
        kf.update(z)

    predictions=np.asarray(predictions).reshape(-1)
    return predictions

def read_data_sets(file_path):
    column_names = ['x-axis', 'y-axis', 'z-axis', 'velocity']
    data = pd.read_csv(file_path,header = None, names = column_names,delimiter='\t')
    return data
        
def filter_noise(data):
    data["x-axis"]=compute_kalman(data["x-axis"])
    data["y-axis"]=compute_kalman(data["y-axis"])
    data["z-axis"]=compute_kalman(data["z-axis"])
    data["velocity"]=compute_kalman(data["velocity"])
    return data

def main():
    for filename in glob.iglob('C:/Users/YeonKyeongNam/Desktop/new/data.txt', recursive = True):
        if os.path.isfile(filename):
            dataset = read_data_sets(file_path = filename)
            dataset.to_csv('C:/Users/YeonKyeongNam/Desktop/new/result.csv', header = None, sep = '\t', encoding = 'utf-8', index =False)
            exit()

main()