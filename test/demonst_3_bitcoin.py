#############################################################
#
# Real-world Data 3. Bitcoin
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

def import_data():
    df1 = pd.read_csv("../data/bitcoin-paper-data/price-data-02-15-2014-06-11-2014.csv")
    df2 = pd.read_csv("../data/bitcoin-paper-data/price-data-11-30-2014-04-28-2015.csv")
    df3 = pd.read_csv("../data/bitcoin-paper-data/price-data-12-17-2015-06-30-2016.csv")

    # df = pd.concat([df1[['timestamp', 'ask', 'bid']], df2[['timestamp', 'ask', 'bid']], df3[['timestamp', 'ask', 'bid']]], axis=0)
    df = df2
    df['time'] = pd.to_datetime(df['timestamp'],unit='s')
    df['price'] = 0.5* (df.ask+df.bid)

    data = df[['timestamp','price']]
    data.index = pd.to_datetime(data['timestamp'], unit='s')
    
    return data

def transform_data(data):
    # z_t = original price observation
    # y_t = difference of z_t
    # x_t = 0 stays the same, -1 decrease, 1 increase
    y_t = data.shift(-1).price - data.price
    x_t = pd.Series(index = y_t.index, dtype=int)
    x_t[y_t == 0] = 0
    x_t[y_t > 0] = 1
    x_t[y_t < 0] = -1
    return x_t, y_t

def plot_data(x_t, y_t, data):
    # histogram
    plt.tight_layout()
    plt.title("Distribution of -1, 0, 1 in X_t")
    plt.hist(x_t, density=True)
    plt.xticks([-1,0,1])
    plt.savefig("result/bitcoin_hist.pdf", format='pdf')
    plt.show()

    # Z_t graph
    plt.tight_layout()
    plt.title("Bitcoin price (Z_t)")
    plt.plot(data.price.values)
    plt.ylim(0,3000)
    plt.xlabel("time")
    plt.ylabel("price")
    plt.xticks()
    plt.savefig("result/bitcoin_z_t.pdf", format='pdf')
    plt.show()

    # X_t graph
    plt.tight_layout()
    plt.title("Bitcoin price change (X_t)")
    plt.scatter(np.arange(100),x_t[:100])
    plt.xlabel("time")
    plt.ylabel("price change")
    plt.savefig("result/bitcoin_x_t.pdf", format='pdf')
    plt.show()



def run_test(data, n, size, samples):
    # input data = (pandas Series)
    # n = number of categories
    verbose = True
    ################
    myRegModel = regModel(n, size, samples)
    myRegModel.fit(plot=False)

    lbs=[]
    for i in np.arange(1,len(data)-size, size):
        uncomp_numbers = data[i:i+size].values
        ratio = lzw_compression_ratio(uncomp_numbers, n)
        ent = myRegModel.get_entropy(ratio, "a multinomial sequence", False)
        lb = h_inverse(ent, n, a=0.001)
        lbs.append(lb)

    plt.title("P(e) changing over time, len={}".format(size))
    plt.plot(lbs)
    plt.axhline(np.mean(lbs), color="red", label="mean={}".format(np.mean(lbs).round(3)))
    plt.xlabel("time")
    plt.ylim(0,0.666)
    plt.ylabel("probability of error")

    plt.legend()
    plt.savefig("result/bitcoin_{}.pdf".format(size), format='pdf')
    plt.show()
    return lbs

def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'figure.autolayout': True})
    data = import_data()
    x_t, y_t = transform_data(data)

    plot_data(x_t, y_t, data)

    
    x_t = x_t[(x_t.index > pd.to_datetime("12/1/2014"))&(x_t.index < pd.to_datetime("12/31/2014"))]
    # val = x_t[(x_t.index > pd.to_datetime("1/1/2015"))&(x_t.index < pd.to_datetime("1/15/2015"))]
    # test = x_t[(x_t.index > pd.to_datetime("1/16/2015"))&(x_t.index < pd.to_datetime("3/30/2015"))]

    
    # change -1 to 2
    x_t[x_t == -1] = 2
    
    n = 3
    samples = 100
    size = 2 ** 14
    # lbs = run_test(x_t, n, size, samples)

    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":

    main()
