#############################################################
#
# Example 3. entropy estimation model test
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

from tslb.src.lzw import *
from tslb.src.utils import *
from tslb.src.regModel import regModel as regModel


def test():
    n = 2
    samples = 100
    size = 1024
    #############

    myRegModel = regModel(n, size, samples)
    myRegModel.fit(plot=False)

    # sample sequence to test - 1. multinomial
    diff_list=[]
    for num in range(100):
        p = random_p(n)
        uncomp_numbers = multinomial(size, p)
        multi_ratio = lzw_compression_ratio(uncomp_numbers, n)
        multi_ent = myRegModel.get_entropy(multi_ratio, "a multinomial sequence", False)
        multi_ent_true = entropy(p)
        diff = multi_ent_true - multi_ent
        diff_list.append(diff)

    plt.hist(diff_list)
    plt.show()

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
