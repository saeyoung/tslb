#############################################################
#
# Log(k*i) graph
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
    k = 3
    i = np.linspace(1,1000000,1000000)
    x = k*i
    y = np.ceil(np.log2(x))

    plt.rcParams.update({'font.size': 16})

    plt.title("Length of W_i when k={}".format(k))
    plt.plot(y)
    plt.ylim(0,25)
    plt.xlabel("i (time index)")
    plt.ylabel("f(i)=ceil(log(k*i))")
    plt.savefig("result/log_graph.pdf", format='pdf')
    plt.show()
    


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
