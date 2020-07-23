#############################################################
#
# TSPDB demonstration
#
#############################################################
import sys, os
sys.path.append("../..")
sys.path.append("..")
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt

from tslb.src.lzw import *
from tslb.src.utils import *
from tslb.src.continuous import *
from tslb.src.regModel import regModel as regModel

plt.rcParams.update({'font.size': 14})

def import_data(string):
	train = pd.read_csv("../data/tspdb_data/{}_train.csv".format(string))
	test = pd.read_csv("../data/tspdb_data/{}_test.csv".format(string))
	return train, test

def lower_bounds(test, k_s=[1,2,3,4,5], plot=False):
	samples = 100

	lbs=[]
	for k in k_s:
		n= 2**k
		#####
		seq = test.y
		size = len(seq)
		#####

		myRegModel = regModel(n, size, samples)
		myRegModel.fit(plot=False)

		# discretize the sequence
		discretized_seq, categories = discretize(seq, n)
		uncomp_numbers = list(discretized_seq)
		ratio = lzw_compression_ratio(uncomp_numbers, n)
		ent = myRegModel.get_entropy(ratio, "a multinomial sequence", plot)
		lb = h_inverse(ent, n, a=0.001)
		lbs.append(lb)
	return lbs

def pred_errors(test, k_s=[1,2,3,4,5]):
	e1=[]
	e2=[]
	e3=[]
	for k in k_s:

		n= 2**k

		seq_test, categories = discretize(test.y, n)

		seq_lstm, categories2 = cut(test.yhat_lstm, categories)
		seq_deepar, categories2 = cut(test.yhat_deepar, categories)
		seq_tspdb, categories2 = cut(test.yhat_tspdb, categories)

		e1.append(get_error(seq_test, seq_lstm))
		e2.append(get_error(seq_test, seq_deepar))
		e3.append(get_error(seq_test, seq_tspdb))
		
	#     print("n = ", n)
	#     print(e1)
	#     print(e2)
	#     print(e3)
	#     print()
	return e1, e2, e3

def experiment(string):
	k_s=[1,2,3,4,5]

	filename = "tspdb_{}".format(string)
	train, test = import_data(string)
	lbs = lower_bounds(test, k_s)
	e1, e2, e3 = pred_errors(test, k_s)

	plt.plot(e1, marker='.', label="lstm", zorder=0)
	plt.plot(e2, marker='.', label="deepAR", zorder=0)
	plt.plot(e3, marker='.', label="tspdb", zorder=0)
	plt.plot(lbs, marker='.', label="lower bound", zorder=10)
	plt.ylim(0,1)
	plt.ylabel("probability of error")
	plt.xticks(np.arange(5),np.arange(1,6))
	plt.xlabel("k, #bins=2^k")
	plt.legend()

	plt.savefig("result/{}.pdf".format(filename), format='pdf')
	plt.clf()
	# plt.show()

def main():
	print("*******************************************************")
	print("*******************************************************")
	print("********** Running the Testing Scripts. ***************")
	
	for string in ["elec","financial","traffic"]:
		print(string)
		experiment(string)

	print("********** Testing Scripts Done. **********************")
	print("*******************************************************")
	print("*******************************************************")

if __name__ == "__main__":

	main()
