from filterpy.kalman import KalmanFilter
import numpy as np

def call_kalman(init_v, velocity):
    kf = KalmanFilter(dim_x=2, dim_z=1)

    # Initial value
    kf.x = np.array([[0], [init_v]])

    # Mathematical system modeling
    kf.F = np.array([[1, 1/30],
                [0, 1]])

    kf.H = np.array([[0., 1.]])

    kf.R = 5000000

    kf.P = 0.000000000001

    kf.Q = 500

    z = np.array([[velocity]])

    kf.predict(u=None, B=None, F=kf.F, Q=kf.Q)
    kf.update(z, kf.R, kf.H)

    return (kf.x[0][0], kf.x[1][0])