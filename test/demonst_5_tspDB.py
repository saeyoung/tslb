#############################################################
#
# Real-world Data 5. tspDB
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
from tslb.src.continuous import *
from tslb.src.regModel import regModel as regModel

def import_data(string):
    train = pd.read_csv("../data/tspdb_data/{}_train.csv".format(string))
    test = pd.read_csv("../data/tspdb_data/{}_test.csv".format(string))
    return train, test

def plot_data(subject, test):
    plt.title("tspDB {} data (test set)".format(subject))
    plt.plot((test.y))
    plt.xlabel("time")
    plt.savefig("result/tspDB_data_{}.pdf".format(subject), format='pdf')
    plt.show()

    plt.title("tspDB {} data, FOD (test set)".format(subject))
    plt.plot(get_diff(test.y))
    plt.xlabel("time")
    plt.savefig("result/tspDB_fod_{}.pdf".format(subject), format='pdf')
    plt.show()

def get_diff(series):
    return (series.shift(-1) - series)[:-1]

def run_test(test, samples):
    lbs=[]
    for k in [1,2,3,4,5]:
        n= 2**k

        #####
        seq = get_diff(test.y)
        size = len(seq)
        #####

        myRegModel = regModel(n, size, samples)
        myRegModel.fit(plot=False)


        # discretize the sequence
        discretized_seq, categories = discretize(seq, n)

        # convert format and get p_tilda
        uncomp_numbers = list(discretized_seq)
        ratio = lzw_compression_ratio(uncomp_numbers, n)
        ent = myRegModel.get_entropy(ratio, "a multinomial sequence", False)
        lb = h_inverse(ent, n, a=0.001)
        lbs.append(lb)
        print("Lower Bound: ", lb)
    return lbs

def get_pred_error(test):
    e1=[]
    e2=[]
    e3=[]
    for k in [1,2,3,4,5]:

        n= 2**k

        seq_test, categories = discretize(get_diff(test.y), n)

        seq_lstm, categories2 = cut(get_diff(test.yhat_lstm), categories)
        seq_deepar, categories2 = cut(get_diff(test.yhat_deepar), categories)
        seq_tspdb, categories2 = cut(get_diff(test.yhat_tspdb), categories)

        e1.append(get_error(seq_test, seq_lstm))
        e2.append(get_error(seq_test, seq_deepar))
        e3.append(get_error(seq_test, seq_tspdb))
    return e1, e2, e3

def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'figure.autolayout': True})


    ls = ["electricity","financial","traffic"]
    subject = ls[2]
    samples = 100
    #####

    train, test = import_data(subject)
    plot_data(subject, test)

    lbs = run_test(test, samples)
    e1, e2, e3 = get_pred_error(test)

    plt.title("Estimated P(e) for varying k (2^k = #bins)")
    plt.plot(e1, label="lstm")
    plt.plot(e2, label="deepAR")
    plt.plot(e3, label="tspdb")
    plt.plot(lbs, marker='.', label="lower bound")
    plt.ylim(0,1)
    plt.xticks(np.arange(5),np.arange(1,6))
    plt.xlabel("k (#bins=2^k)")
    plt.legend()
    plt.savefig("result/tspDB_test_{}.pdf".format(subject), format='pdf')
    plt.show()
        

    
    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":

    main()
