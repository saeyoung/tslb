#############################################################
#
# Test 2-3. Kalman filter
#
#############################################################
import sys, os
sys.path.append("../..")
sys.path.append("..")
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import copy
import pickle
from math import log, e
from sklearn.linear_model import LinearRegression
from numpy.linalg import eig

from utils import *
from continuous import *

from pykalman import KalmanFilter

def test():
    size = 10
    ##########
    # specify parameters
    random_state = np.random.RandomState(0)
    transition_matrix = [[1, 0.1], [0, 1]]
    transition_offset = [-0.1, 0.1]
    observation_matrix = np.eye(2) + random_state.randn(2, 2) * 0.1
    observation_offset = [1.0, -1.0]
    transition_covariance = np.eye(2)
    observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.1
    initial_state_mean = [5, -5]
    initial_state_covariance = [[1, 0.1], [-0.1, 1]]

    # sample from model
    kf = KalmanFilter(
        transition_matrix, observation_matrix, transition_covariance,
        observation_covariance, transition_offset, observation_offset,
        initial_state_mean, initial_state_covariance,
        random_state=random_state
    )
    states, observations = kf.sample(
        n_timesteps=size,
        initial_state=initial_state_mean
    )

    print(states)
    print(states[:,0])


    # # estimate state with filtering and smoothing
    # filtered_state_estimates = kf.filter(observations)[0]
    # smoothed_state_estimates = kf.smooth(observations)[0]

    # # draw estimates
    # plt.figure()
    # lines_true = plt.plot(states, color='b')
    # lines_filt = plt.plot(filtered_state_estimates, color='r')
    # lines_smooth = plt.plot(smoothed_state_estimates, color='g')
    # lines_obs = plt.plot(observations, color="pink")
    # plt.legend((lines_true[0], lines_filt[0], lines_smooth[0], lines_obs[0]),
    #           ('true', 'filt', 'smooth', 'obs'),
    #           loc='lower right'
    # )
    # plt.show()


def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")

    test()

    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":

    main()
