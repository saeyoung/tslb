#############################################################
#
# Test 2-2. Continuous distribution, Gaussian Linear Model
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

def KalmanSequence(size, a):
    # a = transition matrix's (0,0) and (1,1) coordinate
    ##########
    # specify parameters
    random_state = np.random.RandomState(0)
    transition_matrix = [[a, 0], [0, a]]
    transition_offset = [0, 0]
    observation_matrix = np.eye(2) #+ random_state.randn(2, 2) * 0.1
    observation_offset = [0, 0]
    transition_covariance = np.eye(2)
    observation_covariance = np.eye(2) #+ random_state.randn(2, 2) * 0.1
    initial_state_mean = [5, -5]
    initial_state_covariance = [[1, 0], [0, 1]]

    # sample from model
    kf = KalmanFilter(
        transition_matrix, observation_matrix, transition_covariance,
        observation_covariance, transition_offset, observation_offset,
        initial_state_mean, initial_state_covariance,
        random_state=random_state
    )
    states, observations = kf.sample(n_timesteps=size, initial_state=initial_state_mean)

    # estimate state with filtering and smoothing
    filtered_state_estimates = kf.filter(observations)[0]
    # smoothed_state_estimates = kf.smooth(observations)[0]

    return states, observations, filtered_state_estimates

def test(seq, est, n, k, name, plot=False, verbose=True):
    # discretize the sequence
    discretized_seq, categories = discretize(seq, n)
    discretized_est, categories2 = pd.cut(est, categories, labels=np.arange(0, n), retbins=True)
    # print(categories == categories2)

    prediction_error = np.mean(discretized_seq[:-1] != discretized_est[:-1])

    # convert format and get p_tilda
    uncomp_numbers = list(discretized_seq)
    p_tilda = get_p_tilda(uncomp_numbers, n)
    # print(uncomp_numbers)
    # plt.hist(uncomp_numbers)
    # plt.show()

    # make dictionary
    dictionary = {i : chr(i) for i in range(n)}

    # convert number list to string    
    uncompressed = str()        
    for i in uncomp_numbers:
        uncompressed = uncompressed + dictionary[i]

    # compression
    compressed = compress(uncompressed)
    compression_ratio = len(compressed)/len(uncompressed)

    # entropy
    estimated_ent = get_entropy(n, len(uncomp_numbers), compression_ratio, name=name, plot=plot)

    # lower bound
    lb = h_inverse(estimated_ent,n, a=0.005)

    if verbose:
        print("p_tilda            : ", np.round(p_tilda,3))
        print("Compression ratio  : ", compression_ratio)
        print("Error lower bound  : ", lb)
        print("[sanity chk] Est.ent~", h(lb,n))
        print("Estimated entropy  : ", estimated_ent)
        print("Kalman filter error: ", prediction_error)

    return compression_ratio, estimated_ent, lb, prediction_error

def experiment(a):
    #### edit here ####
    size = 1024
    verbose = True
    plot = False
    # title = "Gaussian Linear Model, a={}".format(a)
    title = "Gaussian Linear Model, a={}".format(a)
    path = "result/continuous/dependent_seq"
    ####################

    if not os.path.isdir(path):
        os.makedirs(path)

    # produce a sequence
    # seq = glm(init=0, a=a, length=size)
    states, observations, filtered_state_estimates = KalmanSequence(size, a)

    # draw estimates
    plt.clf()
    plt.title("Gaussian Linear Model, a={}".format(a))
    plt.plot(observations, color="pink", label= 'obs')
    plt.plot(states, color='b', label="true")
    plt.plot(filtered_state_estimates, color='r', label="filter")
    # plt.plot(smoothed_state_estimates, color='g', label="smooth")
    plt.legend()
    plt.savefig("{}/glm_{}_{}_sequence.eps".format(path, a, size), format='eps')
    # plt.show()
    plt.clf()

    seq = states[:,0]
    est = filtered_state_estimates[:,0]

    numbin=[]
    binsize=[]
    y=[]
    z=[]
    for k in range(1,9):
        print(k)
        # dict size
        n = 2**k
        name = "a discretized r.v. into {} buckets".format(n)
        compression_ratio, estimated_ent, lb, prediction_error = test(seq, est, n, k, name, plot, verbose)

        binsize.append(2**(-k))
        numbin.append(n)
        y.append(lb)
        z.append(prediction_error)
        print()

    plt.title(title)
    plt.hist(seq, density=True)
    plt.savefig("{}/glm_{}_{}_hist.png".format(path, a, size), format='eps')
    # plt.show()
    plt.clf()

    plt.title(title)
    plt.plot(numbin,y, marker='.', label="error lower bound")
    plt.plot(numbin,z, marker='.', label="kalman filter error")
    plt.xlabel("# bins (=2^k)")
    plt.ylabel("Misclassification error")
    plt.ylim(0,1)
    plt.legend()
    plt.savefig("{}/glm_{}_{}_error-numbin.eps".format(path, a, size), format='eps')
    # plt.show()
    plt.clf()

    plt.title(title)
    plt.plot(binsize,y, marker='.', label="error lower bound")
    plt.plot(binsize,z, marker='.', label="kalman filter error")
    plt.xlabel("Bin size (=1/2^k)")
    plt.ylabel("Misclassification error")
    plt.ylim(0,1)
    plt.legend()
    plt.savefig("{}/glm_{}_{}_error-binsize.eps".format(path, a, size), format='eps')
    # plt.show()
    plt.clf()

def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")

    for a in [0.1,0.5,1]:
        experiment(a)

    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":

    main()
