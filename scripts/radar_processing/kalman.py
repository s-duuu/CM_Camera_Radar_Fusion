import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import inv
from skimage.metrics import structural_similarity

def kalman_filter(z_meas, x_esti, P):
    """칼만필터 알고리즘 (매개변수 : 측정값, 추정값, 오차공분산)"""
    # (1) Prediction.
    x_pred = A @ x_esti
    P_pred = A @ P @ A.T + Q

    # (2) Kalman Gain.
    K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)

    # (3) Estimation.
    x_esti = x_pred + K @ (z_meas - H @ x_pred)

    # (4) Error Covariance.
    P = P_pred - K @ H @ P_pred

    return x_esti, P

def get_ball_pos(iimg=0):
    
    # Read images.
    imageA = cv2.imread('./data/10.TrackKalman/Img/bg.jpg')
    imageB = cv2.imread('./data/10.TrackKalman/Img/{}.jpg'.format(iimg+1))

    # Convert the images to grayscale.
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # 두 이미지 사이의 구조적 유사성 지수(SSIM) 계산, 차이 이미지가 반환되는지 확인.
    _, diff = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype('uint8') 

    # 서로 다른 두 이미지(배경, 공)의 영역을 구하기 위해 이미지 임계값 지정 후 윤곽선 찾기 
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    xc = int(M['m10'] / M['m00'])  # center of x as true position.
    yc = int(M['m01'] / M['m00'])  # center of y as true position.

    v = np.random.normal(0, 15)  # v: 위치의 측정잡음

    xpos_meas = xc + v  # x_pos_meas: 위치x의 측정값 (observable). 
    ypos_meas = yc + v  # y_pos_meas: 위치y의 측정값 (observable). 

    return np.array([xpos_meas, ypos_meas])

n_samples = 24
dt = 1
A = np.array([[ 1, dt,  0,  0],
              [ 0,  1,  0,  0],
              [ 0,  0,  1, dt],
              [ 0,  0,  0,  1]])
H = np.array([[ 1,  0,  0,  0],
              [ 0,  0,  1,  0]])
Q = 1.0 * np.eye(4)
R = np.array([[50,  0],
              [ 0, 50]])

# Initialization for estimation.
x_0 = np.array([0, 0, 0, 0])  # (x-pos, x-vel, y-pos, y-vel) by definition in book. / 추정값 초기위치
P_0 = 100 * np.eye(4)

xpos_meas_save = np.zeros(n_samples)
ypos_meas_save = np.zeros(n_samples)
xpos_esti_save = np.zeros(n_samples)
ypos_esti_save = np.zeros(n_samples)

x_esti, P = None, None
for i in range(n_samples):
    z_meas = get_ball_pos(i)
    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P = kalman_filter(z_meas, x_esti, P)

    xpos_meas_save[i] = z_meas[0]
    ypos_meas_save[i] = z_meas[1]
    xpos_esti_save[i] = x_esti[0]
    ypos_esti_save[i] = x_esti[2]