#############################################################
#
# Real-world Data 4. NBA
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


def get_first_diff(seq):
    return (seq.shift(-1) - seq).values[:-1].astype(int)

def get_year(df, yr=213):
    return df[(df.nbaId > yr*100000) & (df.nbaId < (yr+1)*100000)]

def get_matrix(df, fr='5S'):
    df_int = df.copy()
    df_int = df_int[df_int.TIME_INT.shift(-1) != df_int.TIME_INT]    # remove the rows with the same TIME_INT

    ### Create Matrix only with Q1-Q4
    # 1230 rows : 1230 games in total
    # 193 columns : 48 mins (4 Quarters) / 15 secs interval

    # only quarter 4
    df_q4 = df_int[df_int.TIME_INT <= pd.to_timedelta("00:48:00")]
    df_q4.loc[:,'TIME_INT'] = pd.to_datetime(df_q4.loc[:,'TIME_INT'])
    # time_index = pd.timedelta_range(start = pd.to_timedelta("00:00:00"), end = pd.to_timedelta("00:48:00"), freq='15s')

    df_q4_home = pd.pivot_table(df_q4, values='HOME_SCORE', columns=['nbaId'],index=['TIME_INT'])
    df_q4_home = df_q4_home.fillna(method = 'ffill')
    df_q4_home = df_q4_home.asfreq(freq=fr, method='ffill')

    df_q4_away = pd.pivot_table(df_q4, values='AWAY_SCORE', columns=['nbaId'],index=['TIME_INT'])
    df_q4_away = df_q4_away.fillna(method = 'ffill')
    df_q4_away = df_q4_away.asfreq(freq=fr, method='ffill')

    df_q4_home = df_q4_home.T
    df_q4_away = df_q4_away.T
    return df_q4_home, df_q4_away

def import_data(fr = '15S', yr = 214):
    df = pd.read_pickle("../data/nba_scores_2103-2018.pkl")

    df_q4_home, df_q4_away = get_matrix(get_year(df, yr=yr), fr=fr)
    df_q4 = pd.concat([df_q4_home, df_q4_away])
    return df_q4

def plot_data():
    df_q4 = import_data(fr = '15S', yr = 214)
    score = df_q4.iloc[0,:]

    plt.title("NBA game score")
    plt.plot(score.values)
    plt.xlabel("time")
    plt.ylabel("score")
    plt.savefig("result/nba_score.pdf", format='pdf')
    plt.show()

    plt.title("NBA game score difference (15s interval)")
    plt.scatter(range(len(get_first_diff(score))),get_first_diff(score))
    plt.xlabel("time")
    plt.ylabel("score")
    plt.savefig("result/nba_score_diff.pdf", format='pdf')
    plt.show()

def year_test(df, fr = '15S', yr = 213):
    #####
    df_q4_home, df_q4_away = get_matrix(get_year(df, yr=yr), fr=fr)

    samples = 100
    size = df_q4_home.shape[1]-1

    myRegModel3 = regModel(3, size, samples)
    myRegModel4 = regModel(4, size, samples)
    myRegModel5 = regModel(5, size, samples)
    myRegModel6 = regModel(6, size, samples)
    myRegModel7 = regModel(7, size, samples)

    myRegModel3.fit(plot=False)
    myRegModel4.fit(plot=False)
    myRegModel5.fit(plot=False)
    myRegModel6.fit(plot=False)
    myRegModel7.fit(plot=False)

    lbs_home=[]
    for i in range(df_q4_home.shape[0]):
        seq = df_q4_home.astype(int).iloc[i]
        uncomp_numbers = get_first_diff(seq)
        n = max(uncomp_numbers)+1

        print(n)    
        if n==3:
            myRegModel = myRegModel3
        elif n==4:
            myRegModel = myRegModel4
        elif n==5:
            myRegModel = myRegModel5
        elif n==6:
            myRegModel = myRegModel6
        elif n==7:
            myRegModel = myRegModel7
            
        if np.sum(uncomp_numbers <0) !=0:
            continue

        ratio = lzw_compression_ratio(uncomp_numbers, n)
        ent = myRegModel.get_entropy(ratio, "a multinomial sequence", False)
        lb = h_inverse(ent, n, a=0.001)
        lbs_home.append(lb)

    lbs_away=[]
    for i in range(df_q4_away.shape[0]):
        seq = df_q4_away.astype(int).iloc[i]
        uncomp_numbers = get_first_diff(seq)
        n = max(uncomp_numbers)+1

        print(n)    
        if n==3:
            myRegModel = myRegModel3
        elif n==4:
            myRegModel = myRegModel4
        elif n==5:
            myRegModel = myRegModel5
        elif n==6:
            myRegModel = myRegModel6
        elif n==7:
            myRegModel = myRegModel7
            
        if np.sum(uncomp_numbers <0) !=0:
            continue
            
        ratio = lzw_compression_ratio(uncomp_numbers, n)
        ent = myRegModel.get_entropy(ratio, "a multinomial sequence", False)
        lb = h_inverse(ent, n, a=0.001)
        lbs_away.append(lb)

    lbs = np.append(np.array(lbs_home), np.array(lbs_away))
    lbs_df = pd.DataFrame(lbs, columns=[yr])
    
    return lbs_df

def save_data():
    df = pd.read_pickle("../data/nba_scores_2103-2018.pkl")
    
    lbs_df_all = pd.DataFrame()
    for yr in [213,214,215,216,217,218]:
    # for yr in [213,214]:
        lbs_df = year_test(df, fr = '15S', yr = yr)
        lbs_df_all = pd.concat([lbs_df_all, lbs_df], axis=1)

    print(lbs_df_all)

    lbs_df_all.to_pickle("lbs_df_all.pkl")

def plot_hist(lbs, year):
    m = round(np.mean(lbs),3)
    plt.title("Error lower bound histogram \n NBA season {}".format(year))
    plt.hist(lbs)
    plt.axvline(np.mean(lbs), color='red', label="mean = {}".format(m))
    plt.xlim(0,1)
    plt.xlabel("classification error")
    plt.legend()
    plt.savefig("result/nba_hist_{}.pdf".format(year), format='pdf')

    plt.show()
    print("mean   : ", np.mean(lbs))
    print("median : ", np.median(lbs))
    print("min    : ", np.min(lbs))
    print("max    : ", np.max(lbs))
    print("std    : ", np.std(lbs))


def main():
    print("*******************************************************")
    print("*******************************************************")
    print("********** Running the Testing Scripts. ***************")
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'figure.autolayout': True})

    # save_data()

    lbs_df_all = pd.read_pickle("lbs_df_all.pkl")

    plot_hist(lbs_df_all[213], 2013)
    plot_hist(lbs_df_all[218], 2018)

    # box plot
    lbs_df_all.columns=["2013","2014","2015","2016", "2017", "2018"]
    plt.title("P(e) distribution per season")
    lbs_df_all.boxplot()
    plt.xlabel("season")
    plt.ylabel("probability of error")
    # plt.ylim(0.1,0.4)
    plt.savefig("result/nba_box.pdf", format='pdf')

    plt.show()

    
    print("********** Testing Scripts Done. **********************")
    print("*******************************************************")
    print("*******************************************************")

if __name__ == "__main__":

    main()
