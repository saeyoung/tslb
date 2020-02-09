#############################################################
#
# Test 2-1. Continuous Distribution, i.i.d. Beta
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

def test(seq, n, k, name, plot, verbose):
    # discretize the sequence
    discretized_seq, categories = discretize(seq, n)

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
        

    return compression_ratio, estimated_ent, lb

def discretized_p(dist, param, n):
    len = 10000000
    samples, categories = discretize(get_sequence(dist, param, len),n)
    p=[]
    for i in range(n):
        p.append(np.mean(samples==i))
    return p

def experiment(param):
    #### edit here ####
    dist = "beta"
    # param = [2,1]
    # dist = "uniform"
    # param = 1
    size = 1000
    verbose = True
    plot = False
    title = "{} distribution ({},{})".format(dist, param[0],param[1])
    path = "result/continuous"
    ####################

    if not os.path.isdir(path):
        os.makedirs(path)

    # produce a sequence
    seq = get_sequence(dist, param, size)

    x=[]
    y=[]
    for k in range(1,9):
        print(k)
        print("{} distribution ({})".format(dist, param))
        # dict size
        n = 2**k
        name = "a discretized uniform r.v. into {} buckets".format(n)
        p = discretized_p(dist, param, n)
        theoretical_ent = entropy(p)
        compression_ratio, estimated_ent, lb = test(seq, n, k, name, plot, verbose)
        x.append(2**(-k))
        y.append(lb)


        print("Discretized entropy: ",theoretical_ent)
        # print("Theoretical p      :",p)
        print()

    plt.title(title)
    plt.hist(seq, density=True)
    plt.savefig("{}/{}_{}_{}_hist.png".format(path, dist, param, size))
    # plt.show()
    plt.clf()

    plt.title(title)
    plt.plot(x,y, marker='.')
    plt.xlabel("k")
    plt.ylabel("Estimated error lower bound")
    plt.ylim(0,1)
    plt.savefig("{}/{}_{}_{}_1.png".format(path, dist, param, size))
    # plt.show()
    plt.clf()


def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")

    param_space=[]
    for i in range(1,3):
        for j in range(1,3):
            param_space.append([i,j])

    for param in param_space:
        experiment(param)

    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":

    main()
