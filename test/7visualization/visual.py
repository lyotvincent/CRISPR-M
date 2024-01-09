import os, time, sys
from turtle import color
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import load_model
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import seaborn as sns
from Bio.SeqUtils import MeltingTemp as mt
from Bio.Seq import Seq

sys.path.append("../../codes")
from positional_encoding import PositionalEncoding
from encoding import encode_by_base_pair_vocabulary, encode_by_base_vocabulary

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
np.random.seed(42)
# plt.style.use("ggplot")

def generate_seq():
    base_dict = {0:"A", 1:"C", 2:"G", 3:"T"}
    randlist = np.random.randint(low=0, high=4, size=(20), dtype=int)
    seq = list()
    for i in range(20):
        seq.append(base_dict[randlist[i]])
    seq = "".join(seq)
    seq += "AGG"
    return seq


def visual(origin_on_target_seq=None, model=None):
    base_dict = {0:"A", 1:"C", 2:"G", 3:"T", 4:"_"}
    # model = load_model('tcrispr_model.h5', custom_objects={"PositionalEncoding": PositionalEncoding})

    m = np.zeros([5, 23], dtype=np.float32)
    # origin_on_target_seq = 'GGCACTGCGGCTGGAGGTGGAGG'
    # origin_off_target_seq = 'GGCACTGCTGCTAGAGGTGCAGG'

    # origin_on_target_seq = 'GGGGGGGGGGGGGGGGGGGGAGG'
    # origin_on_target_seq = 'AAAAAAAAAAAAAAAAAAAAAGG'
    # origin_on_target_seq = 'CCCCCCCCCCCCCCCCCCCCAGG'
    # origin_on_target_seq = 'TTTTTTTTTTTTTTTTTTTTAGG'

    for i in range(23):
        for j in range(5):
            # on_target_seq = list(origin_on_target_seq)
            # on_target_seq[i] = base_dict[j]
            # on_target_seq = "".join(on_target_seq)
            on_target_seq = origin_on_target_seq
            # off_target_seq = list(origin_off_target_seq)
            off_target_seq = list(origin_on_target_seq)
            off_target_seq[i] = base_dict[j]
            off_target_seq = "".join(off_target_seq)
            pair_feature = encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq)
            on_feature = encode_by_base_vocabulary(seq=on_target_seq)
            off_feature = encode_by_base_vocabulary(seq=off_target_seq)
            pair_feature = np.array([pair_feature])
            on_feature = np.array([on_feature])
            off_feature = np.array([off_feature])

            y_pred = model.predict(x=[pair_feature, on_feature, off_feature])
            # print(y_pred)
            m[j][i] = y_pred
            # print(m[0].tolist())
            # print(m[1].tolist())
            # print(m[2].tolist())
            # print(m[3].tolist())
    # print(m)

    on_target_seq = origin_on_target_seq
    off_target_seq = origin_on_target_seq
    pair_feature = encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq)
    on_feature = encode_by_base_vocabulary(seq=on_target_seq)
    off_feature = encode_by_base_vocabulary(seq=off_target_seq)
    pair_feature = np.array([pair_feature])
    on_feature = np.array([on_feature])
    off_feature = np.array([off_feature])
    origin_score = model.predict(x=[pair_feature, on_feature, off_feature])

    m_change = m-origin_score
    # print(m_change)
    return m_change

def generate_tm_score(origin_on_target_seq=None, model=None):
    base_dict = {0:"A", 1:"C", 2:"G", 3:"T"}
    # model = load_model('tcrispr_model.h5', custom_objects={"PositionalEncoding": PositionalEncoding})

    score_sum = 0
    tm_sum = 0

    for i in range(23):
        for j in range(4):
            # on_target_seq = list(origin_on_target_seq)
            # on_target_seq[i] = base_dict[j]
            # on_target_seq = "".join(on_target_seq)
            on_target_seq = origin_on_target_seq
            # off_target_seq = list(origin_off_target_seq)
            off_target_seq = list(origin_on_target_seq)
            off_target_seq[i] = base_dict[j]
            off_target_seq = "".join(off_target_seq)
            tm_sum += get_tm(s=on_target_seq[:-3], c_s=off_target_seq[:-3])
            pair_feature = encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq)
            on_feature = encode_by_base_vocabulary(seq=on_target_seq)
            off_feature = encode_by_base_vocabulary(seq=off_target_seq)
            pair_feature = np.array([pair_feature])
            on_feature = np.array([on_feature])
            off_feature = np.array([off_feature])

            y_pred = model.predict(x=[pair_feature, on_feature, off_feature])
            score_sum += y_pred[0][0]
    return score_sum/92, tm_sum/92

def generate_mismatch_num(origin_on_target_seq=None, model=None):
    base_dict = {0:"A", 1:"C", 2:"G", 3:"T"}
    # model = load_model('tcrispr_model.h5', custom_objects={"PositionalEncoding": PositionalEncoding})

    score_change = [0, 0, 0, 0]

    for i in range(1, 5):
        on_target_seq = origin_on_target_seq
        # off_target_seq = list(origin_off_target_seq)
        off_target_seq = list(origin_on_target_seq)
        positions = np.random.choice(a=20, size=i, replace=False)
        for j in positions:
            while True:
                temp_base = base_dict[np.random.randint(low=0, high=4)]
                if temp_base != off_target_seq[j]:
                    off_target_seq[j] = temp_base
                    break
        off_target_seq = "".join(off_target_seq)
        pair_feature = encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq)
        on_feature = encode_by_base_vocabulary(seq=on_target_seq)
        off_feature = encode_by_base_vocabulary(seq=off_target_seq)
        pair_feature = np.array([pair_feature])
        on_feature = np.array([on_feature])
        off_feature = np.array([off_feature])

        y_pred = model.predict(x=[pair_feature, on_feature, off_feature])
        score_change[i-1] = y_pred[0][0]
    
    on_target_seq = origin_on_target_seq
    off_target_seq = origin_on_target_seq
    pair_feature = encode_by_base_pair_vocabulary(on_target_seq=on_target_seq, off_target_seq=off_target_seq)
    on_feature = encode_by_base_vocabulary(seq=on_target_seq)
    off_feature = encode_by_base_vocabulary(seq=off_target_seq)
    pair_feature = np.array([pair_feature])
    on_feature = np.array([on_feature])
    off_feature = np.array([off_feature])
    origin_score = model.predict(x=[pair_feature, on_feature, off_feature])

    score_change = np.array(score_change)
    score_change -= origin_score[0][0]

    return score_change

def draw_f1(m_change=None):
    m_change = pd.read_csv("pred_matrix_change.csv")
    # print(m_change)
    # print(m_change.iloc[1])
    # print(m_change.iloc[:, 1])
    occlude = (m_change.iloc[0]+m_change.iloc[1]+m_change.iloc[2]+m_change.iloc[3]+m_change.iloc[4])/5

    x = np.array([i for i in range(1,24)])
    y = np.array(occlude)

    # fig, ax = plt.subplots()
    ax = plt.subplot(7, 2, 1)
    # ax.xaxis.set_ticks_position("top")
    for i, y2 in enumerate(y):
        draw_rect(ax, i, y, y2)
    plt.plot([0.5, 23.5], [0, 0], color="grey", linestyle="--", linewidth=0.5)
    plt.plot(x,y)

    # plt.xticks(x, ["1.G-G", "2.G-G", "3.C-C", "4.A-A", "5.C-C", "6.T-T", "7.G-G", "8.C-C", "9.G-T", "10.G-G", "11.C-C", "12.T-T", "13.G-A", "14.G-G", "15.A-A", "16.G-G", "17.G-G", "18.T-T", "19.G-G", "20.G-C", "21.A-A", "22.G-G", "23.G-G"], rotation=270)
    plt.xticks(x, fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    # plt.grid(axis="x", color="grey", linestyle="--", linewidth=0.5)
    plt.ylim(min(occlude), max(occlude))
    plt.xlim(0.5, 23.5)
    plt.xlabel("Base position", fontdict={"fontsize":FONTSIZE})
    plt.ylabel("Substitution score", fontdict={"fontsize":FONTSIZE})
    plt.title("(a) The Substitution Scores for the Base Substitution at Each Position", fontdict={"fontsize":FONTSIZE})
    # plt.show()

def draw_rect(ax, i, y, y2):
    left, right, top, bottom = i+0.5, i+1.5, max(y), min(y)
    # mid = (top+bottom)/2
    color_num = (1-(y2-bottom)/(top-bottom))*3
    # # print(color_num)
    ## plan 1
    # if 2 < color_num <= 3:
    #     red, green, blue = 1, 1, color_num - 2 
    # elif 1 < color_num <= 2:
    #     red, green, blue = 1, color_num - 1, 0
    # elif 0 <= color_num <= 1:
    #     red, green, blue = color_num, 0, 0
    ## plan 2
    if 2 < color_num <= 3:
        red, green, blue = 1, 1, (color_num - 2)*4/5+0.15
    elif 1 < color_num <= 2:
        red, green, blue = 1, color_num - 1, 0.15
    elif 0 <= color_num <= 1:
        red, green, blue = color_num*3/5+0.4, 0, 0.15
        red = np.exp(red-1)
    ## plan 3
    # red, green, blue = color_num/5+0.4, color_num*3/10+0.1, color_num/3
    # if red > 1: red = 1
    # if green > 1: green = 1
    # if blue > 1: blue =1
    # red, green, blue = np.exp(red-1), np.exp(green*2-2), np.exp(blue*3-3)
    print(red, green, blue)
    verts = [(left, bottom), (left, top), (right, top), (right, bottom), (left, bottom),]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor=(red, green, blue), linewidth=0, alpha=0.5)
    ax.add_patch(patch)

def draw_f234(num):
    fig_num = {0:"b", 1:"c", 2:"d", 3:"e", 4:"f"}
    base_name = {0:"adenine", 1:"cytosine", 2:"guanine", 3:"thymine", 4:"indel"}
    m_change = pd.read_csv("pred_matrix_change.csv")
    occlude = m_change.iloc[num]
    c = ["#ED1C24", "#22B14C", "#00A2E8", "#FFC90E", "#FFAEC9"]

    x = np.array([i for i in range(1,24)])
    y = np.array(occlude)

    plt.subplot(7, 2, num*2+3)
    plt.plot([0.5, 23.5], [0, 0], color="grey", linestyle="--", linewidth=0.5)
    plt.bar(x,y,color=c[num])

    plt.xticks(ticks=x, labels=x, fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.grid(axis="x", color="grey", linestyle="--", linewidth=0.5)
    # plt.ylim(np.array(m_change).min(), np.array(m_change).max())
    plt.ylim(min(occlude), max(occlude))
    plt.xlim(0.5, 23.5)
    plt.xlabel("Base position", fontdict={"fontsize":FONTSIZE})
    plt.ylabel("Substitution score", fontdict={"fontsize":FONTSIZE})
    plt.title("(%s) The Substitution Scores for the %s Substitution at Each Position" % (fig_num[num], base_name[num]), fontdict={"fontsize":FONTSIZE})
    # plt.show()

def draw_heatmap():
    m_change = pd.read_csv("pred_matrix_change.csv", nrows=4)
    plt.subplot(7, 2, (13,14))
    ax = sns.heatmap(data=m_change,
                # cmap=sns.cubehelix_palette(as_cmap=True),
                cmap=sns.diverging_palette(10, 220, sep=130, n=30, center="light"),
                center=None,
                linewidths=0.5,
                cbar_kws={
                    # "orientation": "horizontal",
                    # "pad": 0.25
                },
                xticklabels=[i for i in range(1, 24)],
                yticklabels=["A", "C", "G", "T"],
                square=True)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE, rotation=0)
    plt.xlabel("Base", fontdict={"fontsize":FONTSIZE})
    plt.ylabel("Base position", fontdict={"fontsize":FONTSIZE})
    plt.title("(g) The heat map of the off-target effect score changes due to base substitution at each position", fontdict={"fontsize":FONTSIZE})

def draw_gc():
    gc_list = pd.read_csv("gc_list.csv")
    # gc_list = np.array(gc_list)
    plt.subplot(7, 2, (2,4))
    # plt.scatter(gc_list[:, 0], gc_list[:, 1], color="#B97A57", alpha=0.01)
    # sns.scatterplot(x="gc", y="score", data=gc_list, color="#B97A57", alpha=0.01)
    sns.regplot(x="gc", y="score", data=gc_list, order=5, x_jitter=0.02, scatter_kws={"color":"#B97A57", "s":10, "alpha":0.1}, line_kws={"color":"#0080FF", "linewidth":2})
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel("GC content", fontdict={"fontsize":FONTSIZE})
    plt.ylabel("Prediction score", fontdict={"fontsize":FONTSIZE})
    plt.title("(g) The Predicted Results of sgRNA Off-Target Effect in terms of GC Content", fontdict={"fontsize":FONTSIZE})

def draw_tm():
    tm_list = pd.read_csv("tm_list.csv")
    # tm_list = np.array(tm_list)
    plt.subplot(7, 2, (6,8))
    # plt.scatter(tm_list[:, 0], tm_list[:, 1], color="#A349A4", alpha=0.01)
    sns.regplot(x="tm", y="score", data=tm_list, order=5, scatter_kws={"color":"#A349A4", "s":10, "alpha":0.1}, line_kws={"color":"#00EE7F", "linewidth":2})
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel("Melting temperature (Tm)", fontdict={"fontsize":FONTSIZE})
    plt.ylabel("Prediction score", fontdict={"fontsize":FONTSIZE})
    plt.title("(h) The Predicted Results of sgRNA Off-Target Effect in terms of Melting Temperature", fontdict={"fontsize":FONTSIZE})

def draw_mismatch_num():
    mismatch_num_list = pd.read_csv("mismatch_num_list.csv")
    mismatch_num_list = np.array(mismatch_num_list).tolist()
    new_list = list()
    for i in range(len(mismatch_num_list)):
        for j in range(4):
            new_list.append([j+1, mismatch_num_list[i][j]])
    mismatch_num_list = pd.DataFrame(new_list, columns=["mismatch_num", "score"])
    plt.subplot(7, 2, (10,12))
    sns.regplot(x="mismatch_num", y="score", data=mismatch_num_list, order=5, x_jitter=0.25, scatter_kws={"color":"#00C1EE", "s":10, "alpha":0.05}, line_kws={"color":"#FF0000", "linewidth":2})
    plt.xticks([1,2,3,4],fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel("Mismatch number", fontdict={"fontsize":FONTSIZE})
    plt.ylabel("Substitution score", fontdict={"fontsize":FONTSIZE})
    plt.title("(i) The Substitution Scores in terms of Number of Nismatches", fontdict={"fontsize":FONTSIZE})

def generate_m_change_matrix():
    n = 10000
    model = load_model('tcrispr_model.h5', custom_objects={"PositionalEncoding": PositionalEncoding})
    m_change = np.zeros([5, 23], dtype=np.float32)
    tm_list = list()
    gc_list = list()
    mismatch_num_list = list()
    for i in range(n):
        if i % 100 == 0:
            print(i, time.time()-time1)
        seq = generate_seq()
        m_change += visual(seq, model)
        average_pred_score, tm = generate_tm_score(seq, model)
        gc = get_gc_content(seq[:-3])
        tm_list.append([tm, average_pred_score])
        gc_list.append([gc, average_pred_score])
        mismatch_num_list.append(generate_mismatch_num(seq, model))
    m_change /= n
    m_change = pd.DataFrame(m_change, columns=[i for i in range(23)])
    print(m_change.shape)
    m_change.to_csv("pred_matrix_change.csv", index=False)
    tm_list = pd.DataFrame(tm_list, columns=["tm", "score"])
    print(tm_list.shape)
    tm_list.to_csv("tm_list.csv", index=False)
    gc_list = pd.DataFrame(gc_list, columns=["gc", "score"])
    print(gc_list.shape)
    gc_list.to_csv("gc_list.csv", index=False)
    mismatch_num_list = pd.DataFrame(mismatch_num_list, columns=[1,2,3,4])
    print(mismatch_num_list.shape)
    mismatch_num_list.to_csv("mismatch_num_list.csv", index=False)

def dna2rna(s):
    trantab = str.maketrans("ATCG", "UAGC")
    mystring = s
    mystring = mystring.translate(trantab)
    return mystring

def get_tm(s, c_s):
    myseq = Seq(dna2rna(s))
    c_myseq = Seq(c_s)
    return mt.Tm_NN(seq=myseq, c_seq=c_myseq, nn_table=mt.R_DNA_NN1)

def get_gc_content(s):
    return (s.count("G")+s.count("C"))/len(s)


if __name__ == "__main__":
    time1 = time.time()
    # generate_m_change_matrix()

    FONTSIZE = 10
    plt.figure(dpi=300, figsize=(11, 16))
    plt.rc("font", family="Times New Roman")
    params = {"axes.titlesize": FONTSIZE,
              "legend.fontsize": FONTSIZE,
              "axes.labelsize": FONTSIZE,
              "xtick.labelsize": FONTSIZE,
              "ytick.labelsize": FONTSIZE,
              "figure.titlesize": FONTSIZE}
    plt.rcParams.update(params)
    draw_f1()
    for i in range(5):
        draw_f234(i)
    # draw_heatmap()
    draw_gc()
    draw_tm()
    draw_mismatch_num()
    plt.subplots_adjust(hspace=0.6)
    plt.savefig(fname="6visual.svg", format="svg", bbox_inches="tight")
    plt.savefig(fname="6visual.tif", format="tif", bbox_inches="tight")
    plt.savefig(fname="6visual.png", format="png", bbox_inches="tight")
    plt.savefig(fname="6visual.eps", format="eps", bbox_inches="tight")

    print(time.time()-time1)