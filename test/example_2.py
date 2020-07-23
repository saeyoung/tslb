#############################################################
#
# Example 2. entropy estimation
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

# fitted model plots for different sequence length
def test1():
    n = 5
    samples = 100
    #############

    for power in [7,8,9,10]:
        size = 2 ** power
        filename = "ex2_model_fitting_{}".format(size)
        myRegModel = regModel(n, size, samples)
        myRegModel.fit(filename=filename)

# two example dots on the fitted model
def test2(size):
    # size = 1024
    n = 2
    samples = 100
    #############

    myRegModel = regModel(n, size, samples)
    myRegModel.fit()
    
    # blue scatters
    plt.scatter(myRegModel.entropy, myRegModel.ratio, marker='.')
    # orange regression
    plt.plot(myRegModel.reg_inv.predict(np.array(myRegModel.ratio).reshape(-1,1)), myRegModel.ratio, label="regression", color="orange")
    

    # sample sequence to test - 1. multinomial
    # p = random_p(n)
    p = [0.8, 0.2]
    uncomp_numbers = multinomial(size, p)
    multi_ratio = lzw_compression_ratio(uncomp_numbers, n)
    multi_ent = myRegModel.get_entropy(multi_ratio, "a multinomial sequence", False)
    multi_ent_true = entropy(p)

    # sample sequence to test - 2. Markov
    # P = random_P(n)
    P = np.array([[0.7, 0.3], [0.6, 0.4]])
    uncomp_numbers = markov(size, P)
    markov_ratio = lzw_compression_ratio(uncomp_numbers, n)
    markov_ent = myRegModel.get_entropy(markov_ratio, "a Markov process", False)
    markov_ent_true = entropy_rate(P)


    # multi
    plt.axvline(multi_ent, color="grey", alpha=0.5)
    plt.axhline(multi_ratio, color="grey", alpha=0.5)
    plt.scatter(multi_ent, multi_ratio, zorder=10, color="red", label="(multinomial entropy) est={}, true={}".format(multi_ent.round(3), np.round(multi_ent_true,3)))

    # Markov
    plt.axvline(markov_ent, color="grey", alpha=0.5)
    plt.axhline(markov_ratio, color="grey", alpha=0.5)
    plt.scatter(markov_ent, markov_ratio, zorder=10, marker="X", color="red", label="(Markov entropy) est={}, true={}".format(markov_ent.round(3), np.round(markov_ent_true,3)))

    plt.title("Estimated entropy of {} of size {}".format("a sequence", myRegModel.size))
    plt.xlabel("entropy")
    plt.ylabel("compression ratio")
    plt.legend(loc="lower right")
    plt.savefig("result/example_2_{}.pdf".format(size), format='pdf')
    plt.show()

# one example dot on the fitted model, Multinomial and Markov (separately)
def test3(size):
    # size = 1024
    n = 5
    samples = 100
    #############

    myRegModel = regModel(n, size, samples)
    myRegModel.fit()
    
    # sample sequence to test - 1. multinomial
    # p = random_p(n)
    p = [0.1,0.1,0.3,0.4,0.1]

    uncomp_numbers = multinomial(size, p)
    multi_ratio = lzw_compression_ratio(uncomp_numbers, n)
    multi_ent = myRegModel.get_entropy(multi_ratio, "a multinomial sequence", False)
    multi_ent_true = entropy(p)

    # blue scatters
    plt.scatter(myRegModel.entropy, myRegModel.ratio, marker='.')
    # orange regression
    plt.plot(myRegModel.reg_inv.predict(np.array(myRegModel.ratio).reshape(-1,1)), myRegModel.ratio, label="regression", color="orange")

    # multi
    plt.axvline(multi_ent, color="grey", alpha=0.5)
    plt.axhline(multi_ratio, color="grey", alpha=0.5)
    plt.scatter(multi_ent, multi_ratio, zorder=10, color="red", label="(multinomial entropy) est={}, true={}".format(multi_ent.round(3), np.round(multi_ent_true,3)))

    plt.axvline(multi_ent_true, color="red", alpha=0.5)

    plt.title("Estimated entropy of {} of size {}".format("a Multinomial Process (k=5)", myRegModel.size))
    plt.xlabel("entropy")
    plt.ylabel("compression ratio")
    plt.legend(loc="lower right")
    plt.savefig("result/ex2_multi_{}.pdf".format(size), format='pdf')
    plt.show()



    # sample sequence to test - 2. Markov
    # P = random_P(n)
    P = np.array([[0.1,0.2,0.3,0.2,0.2],
                [0.1,0.1,0.3,0.2,0.3],
                [0.5,0.2,0.1,0.1,0.1],
                [0.2,0.5,0.1,0.1,0.1],
                [0.1,0.1,0.5,0.2,0.1]])
    uncomp_numbers = markov(size, P)
    markov_ratio = lzw_compression_ratio(uncomp_numbers, n)
    markov_ent = myRegModel.get_entropy(markov_ratio, "a Markov process", False)
    markov_ent_true = entropy_rate(P)
    
    # blue scatters
    plt.scatter(myRegModel.entropy, myRegModel.ratio, marker='.')
    # orange regression
    plt.plot(myRegModel.reg_inv.predict(np.array(myRegModel.ratio).reshape(-1,1)), myRegModel.ratio, label="regression", color="orange")   

    # Markov
    plt.axvline(markov_ent, color="grey", alpha=0.5)
    plt.axhline(markov_ratio, color="grey", alpha=0.5)
    plt.scatter(markov_ent, markov_ratio, zorder=10, marker="X", color="red", label="(Markov entropy) est={}, true={}".format(markov_ent.round(3), np.round(markov_ent_true,3)))
    
    plt.axvline(markov_ent_true, color="red", alpha=0.5)

    plt.title("Estimated entropy of {} of size {}".format("a Markov Process (k=5)", myRegModel.size))
    plt.xlabel("entropy")
    plt.ylabel("compression ratio")
    plt.legend(loc="lower right")
    plt.savefig("result/ex2_markov_{}.pdf".format(size), format='pdf')
    plt.show()

def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")

    # test1()
    
    # for power in [7,8,9,10]:
    #     size = 2**power
    #     test2(size)


    for power in [7,8,9,10]:
        size = 2**power
        test3(size)

    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":

    main()
