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
    # print("1 mean_fpr length=%s, = %s"%(len(mean_fpr), str(mean_fpr)))

    i = 0
    for j in range(len(fpr_list)):
        fpr, tpr = fpr_list[j], tpr_list[j]
        # print("3", len(fpr), len(tpr), type(fpr))
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plt.plot(fpr, tpr, lw=1, alpha=0.1, color=color, label='ROC fold %d (AUC = %0.5f)' % (i, roc_auc))
        # plt.plot(fpr, tpr, lw=1, alpha=0.05, color=color)

        i += 1
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ## TODO the auc should be compute by new tpr&fpr or mean aucs?
    ## i choose new tpr&fpr, because the curve is drawed by new tpr&fpr, the mean auc should be consistent with the curve.
    plt.plot(mean_fpr, mean_tpr, color=color, label=r'%s (AUC=%0.4f$\pm$%0.4f)' % (model_name, mean_auc, std_auc), lw=2, alpha=.8)
    # plt.plot(mean_fpr, mean_tpr, color=color, label=r'%s (AUC=%0.4f$\pm$%0.4f)' % (model_name, np.mean(aucs), std_auc), lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.2,
    #                 label=r'$\pm$ 1 std. dev.')
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.05)

    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()

# #############################################################################
def draw_mean_prc(path, color, model_name):
    with open(path+"precision_point.csv", "rb") as f:
        precision_point_list = pickle.load(f)
    with open(path+"recall_point.csv", "rb") as f:
        recall_point_list = pickle.load(f)
    # print(np.array(precision_point_list).shape)
    # print(np.array(recall_point_list).shape)

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
    std = mean_precision_list.std(axis=0)

    precision_upper = np.minimum(mean_precision + std, 1)
    precision_lower = mean_precision - std

    # plt.plot(mean_precision, mean_recall, label=f'AUC: {np.mean(mean_average_precision):.2f}', color=color)
    # plt.plot(mean_precision, mean_recall, label=r"%s (AUC=%.2f$\pm$%0.2f)" % (model_name, np.mean(mean_average_precision), np.std(mean_average_precision)), color=color)
    plt.plot(mean_precision, mean_recall, label=r"%s (AUC=%.2f$\pm$%0.2f)" % (model_name, auc(mean_recall, mean_precision), np.std(mean_average_precision)), color=color)
    # plt.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=0.3)

    # plt.plot([0, 1], [1, 0], 'b--')
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.ylabel('Precision (Positive Predictive Value)')
    # plt.xlabel('Recall (True Positive Rate)')
    # plt.title('PR-ROC Curve and PR-AUC')
    # plt.legend(loc='best')
    # plt.show()

def draw_boxplot(data, ymin=0.2, ymax=1.05):
    width = 0.2
    crispr_m = np.array([1,2,3,4,5,6])-width*1.5
    crispr_net = crispr_m+width
    r_crispr = crispr_net+width
    crispr_ip = r_crispr+width

    plt.bar(crispr_m, data["CRISPR-M"], width=width, color="#ED1C24", label="CRISPR-M")
    plt.bar(crispr_net, data["CRISPR-Net"], width=width, color="#FFC90E", label="CRISPR-Net")
    plt.bar(r_crispr, data["R-CRISPR"], width=width, color="#22B14C", label="R-CRISPR")
    plt.bar(crispr_ip, data["CRISPR-IP"], width=width, color="#00A2E8", label="CRISPR-IP")

    plt.xlim(0.4, 6.6)
    plt.ylim(ymin, ymax)
    plt.xticks(np.arange(1,7), ["Accuracy", "Precision", "Recall", "F1 score", "F2 score", "SRCC"])
    for i in range(6):
        plt.text(crispr_m[i], data["CRISPR-M"][i], "%.2f"%data["CRISPR-M"][i], va="bottom", ha="center", rotation=90.)
        plt.text(crispr_net[i], data["CRISPR-Net"][i], "%.2f"%data["CRISPR-Net"][i], va="bottom", ha="center", rotation=90.)
        plt.text(r_crispr[i], data["R-CRISPR"][i], "%.2f"%data["R-CRISPR"][i], va="bottom", ha="center", rotation=90.)
        plt.text(crispr_ip[i], data["CRISPR-IP"][i], "%.2f"%data["CRISPR-IP"][i], va="bottom", ha="center", rotation=90.)
    plt.legend(loc="best")

if __name__ == "__main__":
    # draw_mean_roc(path="./CRISPR-M/m81212_n13/1LOGOCV/", color="#ED1C24")
    path_list = ["./CRISPR-Net/1LOGOCV/", "./R-CRISPR/1LOGOCV/", "./CRISPR-IP/1LOGOCV/", "./CRISPR-M/m81212_n13/1LOGOCV/"]
    path2_list = ["./CRISPR-Net/2CIRCLE_GUIDE/", "./R-CRISPR/2CIRCLE_GUIDE/", "./CRISPR-IP/2CIRCLE_GUIDE/", "./CRISPR-M/m81212_n13/2CIRCLE_GUIDE/"]
    model_name_list = ["CRISPR-Net", "R-CRISPR", "CRISPR-IP", "CRISPR-M"]
    c = ["#FFC90E", "#22B14C", "#00A2E8", "#ED1C24", "#FFAEC9"]
    FONTSIZE = 10
    plt.figure(dpi=300, figsize=(9, 17))
    # plt.style.use("fast")
    plt.rc("font", family="Times New Roman")
    params = {"axes.titlesize": FONTSIZE,
              "legend.fontsize": FONTSIZE,
              "axes.labelsize": FONTSIZE,
              "xtick.labelsize": FONTSIZE,
              "ytick.labelsize": FONTSIZE,
              "figure.titlesize": FONTSIZE,
              "font.size": FONTSIZE}
    plt.rcParams.update(params)
    # print(plt.rcParams.keys())

    plt.subplot(4, 2, 1)
    for i in range(4):
        draw_mean_roc(path_list[i], c[i], model_name_list[i])

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='grey', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('(a) LOGOCV: Mean Receiver Operating Characteristic Curve')
    plt.legend(loc="best")

    plt.subplot(4, 2, 2)
    for i in range(4):
        draw_mean_prc(path_list[i], c[i], model_name_list[i])

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('(b) LOGOCV: Mean Precision-Recall Curve')
    plt.legend(loc="best")

    plt.subplot(4, 2, (3,4))
    data = dict()
    data["CRISPR-IP"] = [0.9887638633536042, 0.6259300894964073, 0.3273792330009765, 0.3870632524750847, 0.34490367238983516, 0.42110734962851915]
    data["CRISPR-Net"] = [0.9877456628646859, 0.6006973447114243, 0.35521524397352866, 0.3821735789575994, 0.3546730983867371, 0.41944785658627337]
    data["R-CRISPR"] = [0.987815268180152, 0.6644903896116109, 0.26003627548274777, 0.2868816192268335, 0.2597863271277955, 0.35081020860245243]
    data["CRISPR-M"] = [0.9894828090074835, 0.6696199109801636, 0.3527499850418886, 0.4132518081954994, 0.3668792655557376, 0.4526663520487415]
    draw_boxplot(data, ymin=0.2, ymax=1.06)
    plt.title('(c) LOGOCV: Performance Evaluation on Several Metrics')

    plt.subplot(4, 2, 5)
    for i in range(4):
        draw_mean_roc(path2_list[i], c[i], model_name_list[i])

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='grey', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('(d) CIECLE_GUIDE: Mean Receiver Operating Characteristic Curve')
    plt.legend(loc="best")

    plt.subplot(4, 2, 6)
    for i in range(4):
        draw_mean_prc(path2_list[i], c[i], model_name_list[i])

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('(e) CIECLE_GUIDE: Mean Precision-Recall Curve')
    plt.legend(loc="best")

    plt.subplot(4, 2, (7,8))
    data = dict()
    data["CRISPR-IP"] = [0.9967447752333676, 0.046418274976154576, 0.548, 0.08462510322285006, 0.16821066998660894, 0.1555807797155057]
    data["CRISPR-Net"] = [0.9970467389322826, 0.05396569245485287, 0.654, 0.0991762306396709, 0.20003852907812428, 0.18554579381932795]
    data["R-CRISPR"] = [0.9969696119813213, 0.05865615787569758, 0.672, 0.10672058096979634, 0.21119713229917908, 0.19411038179686932]
    data["CRISPR-M"] = [0.9972584874703763, 0.5962376602526593, 0.394, 0.23207363595179192, 0.2151656038988377, 0.32647996692095604]
    draw_boxplot(data, ymin=0.0, ymax=1.08)
    plt.title('(f) CIECLE_GUIDE: Performance Evaluation on Several Metrics')

    plt.savefig(fname="1indel.svg", format="svg", bbox_inches="tight")
    plt.savefig(fname="1indel.tif", format="tif", bbox_inches="tight")
    plt.savefig(fname="1indel.png", format="png", bbox_inches="tight")
    plt.savefig(fname="1indel.eps", format="eps", bbox_inches="tight")