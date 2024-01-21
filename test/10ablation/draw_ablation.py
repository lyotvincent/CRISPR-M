import numpy as np
from numpy import interp
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import auc


# #############################################################################
def draw_mean_roc(path, color, model_name):
    with open(path+"fpr.csv", "rb") as f:
        fpr_list = pickle.load(f)
    with open(path+"tpr.csv", "rb") as f:
        tpr_list = pickle.load(f)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for j in range(len(fpr_list)):
        fpr, tpr = fpr_list[j], tpr_list[j]
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        i += 1

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ## TODO the auc should be compute by new tpr&fpr or mean aucs?
    ## i choose new tpr&fpr, because the curve is drawed by new tpr&fpr, the mean auc should be consistent with the curve.
    plt.plot(mean_fpr, mean_tpr, color=color, label=r'%s (AUC=%0.4f$\pm$%0.4f)' % (model_name, mean_auc, std_auc), lw=2, alpha=.8)


# #############################################################################
def draw_mean_prc(path, color, model_name):
    with open(path+"precision_point.csv", "rb") as f:
        precision_point_list = pickle.load(f)
    with open(path+"recall_point.csv", "rb") as f:
        recall_point_list = pickle.load(f)

    mean_precision_list = list()
    mean_average_precision = list()
    mean_recall = np.linspace(0, 1, 101)

    for j in range(len(precision_point_list)):

        precision, recall = precision_point_list[j], recall_point_list[j]
        # print(j, precision.shape,recall.shape)
        
        average_precision = auc(recall, precision)
        mean_average_precision.append(average_precision)

        # plt.plot(precision, recall, color=color, alpha=0.05)

        precision = np.interp(mean_recall, precision, recall)
        # precision[0] = 0.0
        mean_precision_list.append(precision)

    mean_precision_list = np.array(mean_precision_list)
    mean_precision = mean_precision_list.mean(axis=0)

    plt.plot(mean_precision, mean_recall, label=r"%s (AUC=%.2f$\pm$%0.2f)" % (model_name, auc(mean_recall, mean_precision), np.std(mean_average_precision)), color=color)




if __name__ == "__main__":
    path_list = ["./without_CNN/", "./without_LSTM/", "./without_Dense/", "./full_model/"]
    path2_list = ["./m81212_n13_without_branch12/", "./m81212_n13_without_branch34/", "./full_model/"]
    # model_name_list = ["without_CNN", "without_LSTM", "without_Dense", "full_model"]
    model_name_list = ["Ablation model 1", "Ablation model 2", "Ablation model 3", "CRISPR-M"]
    model_name_list2 = ["Ablation model a", "Ablation model b", "CRISPR-M"]
    c = ["#FFC90E", "#22B14C", "#00A2E8", "#ED1C24", "#FFAEC9"]
    FONTSIZE = 10
    plt.figure(dpi=300, figsize=(9, 9))
    # plt.style.use("fast")
    plt.rc("font", family="Times New Roman")
    params = {"axes.titlesize": FONTSIZE,
              "legend.fontsize": 9,
              "axes.labelsize": FONTSIZE,
              "xtick.labelsize": FONTSIZE,
              "ytick.labelsize": FONTSIZE,
              "figure.titlesize": FONTSIZE,
              "font.size": FONTSIZE}
    plt.rcParams.update(params)

    plt.subplot(2, 2, 1)
    for i in range(4):
        draw_mean_roc(path_list[i], c[i], model_name_list[i])

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='grey', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('(a) Module Ablation: Mean Receiver Operating Characteristic Curve')
    plt.legend(loc="best")

    plt.subplot(2, 2, 2)
    for i in range(4):
        draw_mean_prc(path_list[i], c[i], model_name_list[i])

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('(b) Module Ablation: Mean Precision-Recall Curve')
    plt.legend(loc="best")

    # plt.subplot(2, 2, 3)
    # for i in range(3):
    #     draw_mean_roc(path2_list[i], c[i+1], model_name_list2[i])

    # plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='grey', alpha=.8)
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate (FPR)')
    # plt.ylabel('True Positive Rate (TPR)')
    # plt.title('(c) Branch Ablation: Mean Receiver Operating Characteristic Curve')
    # plt.legend(loc="best")

    # plt.subplot(2, 2, 4)
    # for i in range(3):
    #     draw_mean_prc(path2_list[i], c[i+1], model_name_list2[i])

    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.ylabel('Precision')
    # plt.xlabel('Recall')
    # plt.title('(d) Branch Ablation: Mean Precision-Recall Curve')
    # plt.legend(loc="best")

    # plt.savefig(fname="10ablation.svg", format="svg", bbox_inches="tight")
    # plt.savefig(fname="10ablation.tif", format="tif", bbox_inches="tight")
    plt.savefig(fname="10ablation.png", format="png", bbox_inches="tight")
    # plt.savefig(fname="10ablation.eps", format="eps", bbox_inches="tight")

