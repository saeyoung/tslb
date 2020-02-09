#############################################################
#
# Example 1. LZW compression
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
    size = 100
    n = 10
    p = random_p(n=10)

    uncomp_numbers = multinomial(size, p)

    print(lzw_compression_ratio(uncomp_numbers, n))

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
