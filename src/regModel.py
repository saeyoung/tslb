######################################################
#
# Regression Model
#
######################################################
import numpy as np
import pandas as pd
import random
import copy
import pickle
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy.linalg import eig

# for entropy
# import zlib
import re
from math import log, e
from io import StringIO

from tslb.src.lzw import *
from tslb.src.utils import *


class regModel():
    # n = alphabet size
    # size = sequence length
    # samples = number of samples to collect
    def __init__(self, n, size, samples):
        self.n = n
        self.size = size
        self.samples = samples

        # ratio = ratio list to fit reg
        # entropy = entropy list to fit entropy
        self.ratio = None
        self.entropy = None
        self.reg = None
        self.reg_inv = None

    def fit(self, verbose=False, plot=True, filename="example"):
        dictionary = {i : chr(i) for i in range(self.n)}
        
        ratio_list =[]
        true_entropy = []
        
        probabilities=[]
        for i in range(self.samples):
            probabilities.append(random_p(self.n))
                
        for p in probabilities:
            true_entropy.append(entropy(p))
            
            uncompressed =str()        
            uncomp_numbers = multinomial(self.size, p)
            for i in uncomp_numbers:
                uncompressed = uncompressed + dictionary[i]

            compressed = compress(uncompressed)
            compression_ratio = len(compressed)/len(uncompressed)
            ratio_list.append(compression_ratio)

        self.ratio = ratio_list
        self.entropy = true_entropy

        # linear regression
        self.reg = LinearRegression(fit_intercept=True).fit(np.array(true_entropy[:]).reshape(-1, 1), np.array(self.ratio[:]))
        self.reg_inv = LinearRegression(fit_intercept=True).fit(np.array(self.ratio[:]).reshape(-1, 1), np.array(true_entropy[:]))
        score = self.reg.score(np.array(true_entropy[:]).reshape(-1, 1), np.array(self.ratio[:])).round(3)

        if verbose:
            print("y = ax + b model")
            print("a = ", self.reg.coef_)
            print("b = ", self.reg.intercept_)

        if plot:
            plt.scatter(self.entropy, self.ratio, marker='.', label = "Compressed samples")
            plt.plot(self.entropy, self.reg.predict(np.array(self.entropy).reshape(-1,1)), label="regression, r2_score={}".format(score), color="orange")
            plt.title("Compression ratio - entropy regression model \n Multinomial with {} states, size={}".format(self.n, self.size))
            plt.xlabel("entropy")
            plt.ylabel("compression ratio")
            plt.legend(loc="lower right")
            plt.savefig("result/{}.pdf".format(filename), format='pdf')
            plt.show()


    def get_entropy(self, compression_ratio, name ="a sequence", plot=True):
        # mapping compression ratio to entropy
        ent = self.reg_inv.predict(np.array(compression_ratio).reshape(-1, 1))[0]

        if plot:
            print("plot")
            print("estimated entropy = ", ent)
            print("compression ratio = ", compression_ratio)
            print("reg.predict(est)  = ", self.reg.predict([[ent]]))
            plt.scatter(self.entropy, self.ratio, marker='.')
            plt.plot(self.reg_inv.predict(np.array(self.ratio).reshape(-1,1)), self.ratio, label="regression", color="orange")
            plt.axvline(ent, color="grey", alpha=0.5)
            plt.axhline(compression_ratio, color="grey", alpha=0.5)
            plt.scatter(ent, compression_ratio, color="red", label="Estimated entropy={}".format(ent.round(3)))

            plt.title("Estimated entropy of {} of size {}".format(name, self.size))
            plt.xlabel("entropy")
            plt.ylabel("compression ratio")
            plt.legend()
            plt.savefig("result/example.pdf", format='pdf')
            plt.show()
        return ent
