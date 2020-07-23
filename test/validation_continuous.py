#############################################################
#
# Validation - continutous distribution
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
from math import log, e, pi
from sklearn.linear_model import LinearRegression
from numpy.linalg import eig
from pykalman import KalmanFilter

from sklearn.neighbors import KernelDensity

from tslb.src.lzw import *
from tslb.src.utils import *
from tslb.src.regModel import regModel as regModel

def KalmanSequence(size, a, rand):
    # X_{t+1} = a*X_t + noise
    ##########
    # specify parameters
    random_state = np.random.RandomState(rand)
    transition_matrix = [[a]]
    transition_offset = [0]
    observation_matrix = np.eye(1) #+ random_state.randn(2, 2) * 0.1
    observation_offset = [0]
    transition_covariance = np.eye(1)
    observation_covariance = np.eye(1) #+ random_state.randn(2, 2) * 0.1
    initial_state_mean = [0]
    initial_state_covariance = [[1]]

    # for 2-dim
    # random_state = np.random.RandomState(0)
    # transition_matrix = [[a, 0], [0, a]]
    # transition_offset = [0, 0]
    # observation_matrix = np.eye(2) #+ random_state.randn(2, 2) * 0.1
    # observation_offset = [0, 0]
    # transition_covariance = np.eye(2)
    # observation_covariance = np.eye(2) #+ random_state.randn(2, 2) * 0.1
    # initial_state_mean = [5, -5]
    # initial_state_covariance = [[1, 0], [0, 1]]

    # sample from model
    kf = KalmanFilter(
        transition_matrix, observation_matrix, transition_covariance,
        observation_covariance, transition_offset, observation_offset,
        initial_state_mean, initial_state_covariance,
        random_state=random_state
    )
    states, observations = kf.sample(n_timesteps=size, initial_state=initial_state_mean)

    # estimate state with filtering and smoothing
    filtered_state_estimates = kf.filter(states)[0]
    # filtered_state_estimates = kf.filter(observations)[0]
    # smoothed_state_estimates = kf.smooth(observations)[0]

    return states, observations, filtered_state_estimates

def window_stack(a, stepsize=1, width=3):
    n = a.shape[0]
    return np.hstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )

def produce_blocks(seq, p):
    # seq (2d array): time series data
    # p (int): the number of previous observations to keep
    X = seq[p:,0:1]
    Y = window_stack(seq,1,p)[:-1,:]
    return X, Y

    
def generate():
    dist = "beta"
    param = [2,1]
    # dist = "uniform"
    # param = 1
    size = 1000
    ##################

    # produce a sequence
    seq = get_sequence(dist, param, size)

    # discretize the sequence
    discretized_seq, categories = discretize(seq, n)

def func(h):
    return (e**h)/(2*pi*e)

def mse(true, est):
    return np.mean((true-est)**2)

def test():
    size=100
    a=0.5
    p=2
    rand=0

    # generate sequence
    states, observations, filtered_state_estimates = KalmanSequence(size, a, rand)

    # # plot sequence
    # plt.plot(states, marker='.', label="true")
    # # plt.plot(observations, label="obs")
    # plt.plot(filtered_state_estimates, marker='.', label="est")
    # plt.legend()
    # plt.show()
    # plt.clf()

    # produce blocks (X:label, Y:features)
    X,Y = produce_blocks(states, p)
    XY = np.concatenate((X,Y), axis=1)
    # print("data")
    # print(states)
    # print(X)
    # print(Y)

    # estimate pdf
    # Compute the total log probability density under the model.
    # aka score = log-likelihood
    kde_x = KernelDensity(kernel='gaussian', bandwidth=2).fit(X)
    kde_y = KernelDensity(kernel='gaussian', bandwidth=2).fit(Y)
    kde_xy = KernelDensity(kernel='gaussian', bandwidth=2).fit(XY)


    print("estimation")
    print(e ** kde_x.score(X))
    print(e ** kde_y.score(Y))
    print(e ** kde_xy.score(XY))


    entropy_est = -np.mean(kde_xy.score(XY) - kde_y.score(Y))
    print("Estimated Lower Bound: ", func(entropy_est))
    print("Kalman Filter MSE    : ", mse(states, filtered_state_estimates))


# def test():
#     n = 2
#     samples = 100
#     size = 1024
#     #############

#     myRegModel = regModel(n, size, samples)
#     myRegModel.fit(plot=False)

    # # sample sequence to test - 1. multinomial
    # diff_list=[]
    # for num in range(100):
    #     p = random_p(n)
    #     uncomp_numbers = multinomial(size, p)
    #     multi_ratio = lzw_compression_ratio(uncomp_numbers, n)
    #     multi_ent = myRegModel.get_entropy(multi_ratio, "a multinomial sequence", False)
    #     multi_ent_true = entropy(p)
    #     diff = multi_ent_true - multi_ent
    #     diff_list.append(diff)

    # plt.hist(diff_list)
    # plt.show()

    # # sample sequence to test - 2. Markov
    # diff_list=[]
    # for num in range(100):
    #     P = random_P(n)
    #     uncomp_numbers = markov(size, P)
    #     markov_ratio = lzw_compression_ratio(uncomp_numbers, n)
    #     markov_ent = myRegModel.get_entropy(markov_ratio, "a Markov process", False)
    #     markov_ent_true = entropy_rate(P)
    #     diff = markov_ent_true - markov_ent
    #     diff_list.append(diff)
        
    # plt.hist(diff_list)
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
