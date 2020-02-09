#############################################################
#
# Test 1. Synthetic Data - Discrete Case
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

def get_string(uncomp_numbers, n):
    # max(n) = 256

    # make dictionary
    dictionary = {i : chr(i) for i in range(n)}

    # convert number list to string
    uncompressed = str()        
    for i in uncomp_numbers:
        uncompressed = uncompressed + dictionary[i]

    return uncompressed

def test(n, uncompressed, size, name, plot):
    # compression
    compressed = compress(uncompressed)
    compression_ratio = len(compressed)/len(uncompressed)
    # print("compression ratio: ",compression_ratio)

    # entropy
    estimated_ent = get_entropy(n, size, compression_ratio, name=name, plot=plot)

    # lower bound
    lb = h_inverse(estimated_ent, n, a=0.005)

    return compression_ratio, estimated_ent, lb

def experiment(dist, n, power, size, plot, verbose, samples): # simple experiment
    theo_ent=[]
    est_ent=[]
    lbs=[]
    for num in range(samples):
        if dist == "multinomial":
            name = "Multinomial process with {} states".format(n)
            # p = random_p(n=n)
            p = [0.4, 0.3, 0.1, 0.1, 0.1]
            uncomp_numbers = multinomial(size, p)
            theoretical_ent = entropy(p)

        elif dist == "markov":
            name = "Markov process with {} states".format(n)
            P = get_P(n=n)
            uncomp_numbers = markov(size, P, initial = 0)
            theoretical_ent = entropy_rate(P)

        p_tilda = get_p_tilda(uncomp_numbers, n)
        uncompressed = get_string(uncomp_numbers, n)

        compression_ratio, estimated_ent, lb = test(n, uncompressed, size, name, plot)
        theo_ent.append(theoretical_ent)
        est_ent.append(estimated_ent)
        lbs.append(lb)

        if verbose:
            print("compression ratio   : ", compression_ratio)
            print("theoretical entropy : ", theoretical_ent)
            print("estimated entropy   : ", estimated_ent)
            print("estimated lb        : ", lb)
            print("p_tilda            : ", np.round(p_tilda,3))
            print()

    error = np.array(est_ent) - np.array(theo_ent)    
    print("mean absolute error : ", np.mean(error))

    return theo_ent, est_ent, lbs, name


def experiment_1(): # error histogram
    #### edit here ####
    dist = "markov"
    power = 10 # size = 2^power
    size = 2 ** power
    plot = False
    verbose = True
    samples = 100
    ###################
    for n in [5,10]:
        theo_ent, est_ent, lbs, name = experiment(dist, n, power, size, plot, verbose, samples)
        error = np.array(theo_ent)-np.array(est_ent)
        plt.title("Theoretical entropy - estimated entropy distribution \n {}".format(name))
        plt.hist(error)
        plt.axvline(np.mean(error), color="red", label="mean={}".format(np.mean(error).round(3)))
        plt.legend()
        plt.savefig("result/err_dist/{}_states_{}_error_distribution_{}.pdf".format(n, dist, size), format='pdf')
        plt.show()
        plt.clf()

def experiment_2(powers):
    #### edit here ####
    dist = "multinomial"
    plot = False
    verbose = True
    samples = 100
    ###################

    # n = the number of states
    for n in [5]:
        data = np.zeros([1,samples])
        for power in powers:
            size = 2 ** power
            theo_ent, est_ent, lbs, name = experiment(dist, n, power, size, plot, verbose, samples)
            error = np.array(est_ent) - np.array(theo_ent)
            data = np.vstack((data, error))
        data = data[1:,:]
        data = data.T #(samples x powers)
        data_abs = np.abs(data)

        # histogram
        for k in range(len(powers)):
            avg = np.mean(data[:,k])
            std = np.std(data[:,k])
            plt.title("Estimated Entropy - Theoretical Entropy \n {}, length=2^{}".format(name,powers[k]))
            plt.hist(data[:,k])
            plt.axvline(avg, color='red', label="mean={}, std={}".format(avg.round(3), std.round(3)))
            plt.legend()
            plt.savefig("result/hist/{}/{}_states_{}_err_distribution_{}_samples_{}.pdf".format(dist, n, dist, samples, powers[k]), format='pdf')
            # plt.show()
            plt.clf()

            print("size = ", 2 ** powers[k])
            print("var  = ", np.var(data[:,k]))
            print("std  = ", np.std(data[:,k]))

        # regression line
        y = np.median(np.log2(data_abs),axis=0).reshape(len(powers),1)
        X = np.arange(len(powers)).reshape(-1, 1)
        reg = LinearRegression().fit(X, y)

        plt.title("Absolute discrepancy between theoretical and estimated entropy \n {}".format(name))
        plt.xlabel("log length (base=2)")
        plt.ylabel("absolute error")
        plt.boxplot(data_abs)
        plt.xticks(np.arange(1,len(powers)+1),powers)
        plt.savefig("result/boxplot/{}/{}_states_{}_boxplot_semilog_{}_samples_{}.pdf".format(dist,n, dist, samples, powers[0]), format='pdf')
        # plt.show()
        plt.clf()

        plt.title("Absolute discrepancy between theoretical and estimated entropy \n {}".format(name))
        plt.xlabel("log length (base=2)")
        plt.ylabel("log absolute error")
        plt.boxplot(np.log2(data_abs))
        plt.xticks(np.arange(1,len(powers)+1),powers)
        plt.savefig("result/boxplot/{}/{}_states_{}_boxplot_loglog_{}_samples_{}.pdf".format(dist,n, dist, samples,powers[0]), format='pdf')
        plt.plot(np.arange(1,len(powers)+1),reg.predict(X), color="red", label="r2 score={}".format(reg.score(X,y).round(3)))
        plt.legend()
        plt.savefig("result/boxplot/{}/{}_states_{}_boxplot_loglog_{}_samples_regression_{}.pdf".format(dist,n, dist, samples,powers[0]), format='pdf')
        # plt.show()
        plt.clf()

def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")

    #### for experiment() ####
    #### edit here ####
    dist = "multinomial"
    n = 5 # number of states
    power = 13 # size = 2^power
    size = 2 ** power
    plot = True
    verbose = True
    samples = 1
    ################### 
    experiment(dist, n, power, size, plot, verbose, samples)

    #### for experiment_1() ####
    # experiment_1()

    #### for experiment_2() ####    
    # powers = [6,7,8,9,10]
    # experiment_2(powers)

    # powers_short = [7,8,9,10]
    # powers_long = [11,12,13,14]
    # experiment_2(powers_short)
    # experiment_2(powers_long)

    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":

    main()
