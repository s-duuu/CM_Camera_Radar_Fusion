import numpy as np

def call_1dkalman(kf, init_v, velocity):
    
    # Initial value
    kf.x = np.array([[0], [init_v]])

    # Mathematical system modeling
    kf.F = np.array([[1, 1/30],
                [0, 1]])

    kf.H = np.array([[0., 1.]])

    kf.R = 100000

    kf.P = 100

    kf.Q = 5000

    z = np.array([[velocity]])

    kf.predict(u=None, B=None, F=kf.F, Q=kf.Q)
    kf.update(z, kf.R, kf.H)

    return kf.x[1][0]

def call_2dkalman(kf, init_d, init_v, distance, velocity):

    kf.x = np.array([[init_d], [init_v]])

    kf.F = np.array([[1, 1/30],
                [0, 1]])
    
    kf.H = np.array([[1., 0.], [0., 1.]])

    kf.R = np.array([[1., 0.], [0., 50000]])
    
    kf.P = np.array([[1., 0.], [0., 1000000]])

    kf.Q = np.array([[1., 0.], [0., 1.]])

    z = np.array([[distance], [velocity]])

    kf.predict(u=None, B=None, F=kf.F, Q=kf.Q)
    kf.update(z, kf.R, kf.H)

    return kf.x