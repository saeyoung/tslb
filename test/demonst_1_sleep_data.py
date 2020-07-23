#############################################################
#
# Real-world Data 1. Sleep Data
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

def import_data():
    df = pd.read_csv("../data/sleep_data.csv", header=None)
    df = df[0].str.split(" ", expand = True)
    df.columns = ["heart_rate", "sleep", "temperature"]
    return df

def plot_data(data):
    plt.title("Sleep Data")
    # plt.scatter(np.arange(len(data)),data, marker='.')
    plt.plot(data, marker='.')
    plt.xlabel("time")
    plt.ylabel("sleep status")
    plt.yticks([1,2,3,4])
    plt.savefig("result/sleep_data.pdf", format='pdf')
    plt.show()


def run_test(data, n):
    # input data = (pandas Series)
    # n = number of categories
    size = len(data)
    verbose = True
    ################

    unique = np.unique(data)

    label = 0
    for x in unique:
        data[data == x] = label
        label += 1
    print(data)

    # observation analysis
    p_tilda=[]
    for i in range(1,n+1):
        p = np.mean(np.array(data.astype(int)) == i)
        p_tilda.append(p)
    print(p_tilda)

    # compression
    uncompressed = list_to_string(list(data))
    compressed = compress(uncompressed)
    compression_ratio = len(compressed)/len(uncompressed)

    # entropy
    estimated_ent = get_entropy(n, size, compression_ratio, name = "sleep data", plot=True)
    # empirical_ent = entropy(p_tilda)
    # empirical_ent = 0

    # lower bound
    lb = h_inverse(estimated_ent, n, a=0.0001)

    if verbose:
        print("p_tilda            : ", np.round(p_tilda,3))
        print("Compression ratio  : ", compression_ratio)
        print("Estimated entropy  : ", estimated_ent)
        # print("Empirical entropy  : ", empirical_ent)
        print("P(e) lower bound   : ", lb)

    # return compression_ratio, estimated_ent, empirical_ent

def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")
    
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'figure.autolayout': True})

    df = import_data()
    data = df.sleep
    n = len(np.unique(data))
    print("n=", n)
    data = data.astype(int)

    plot_data(data)
    # run_test(data, n)

    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":

    main()
