import os, sys
from tkinter import font
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../../1indel/")
from mean_roc_prc import draw_mean_roc, draw_mean_prc

PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def draw_encoding_test():
    df = pd.read_excel(PATH+r'/2encoding_test/fig2/AUPRC.xlsx')
    # print(df)
    crispr_m_6_data = df["CRISPR-M 6 channel encoding"].tolist()
    crispr_ip_data = df["CRISPR-IP"].tolist()
    crispr_ip_without_pam_data = df["CRISPR-IP without PAM channel"].tolist()
    crispr_net_data = df["CRISPR-Net"].tolist()
    crispr_net_with_pam_data = df["CRISPR-Net with PAM channel"].tolist()
    crispr_m_word_embedding_with_PAM_data = df["CRISPR-M word embedding with PAM channel"].tolist()
    crispr_m_word_embedding_data = df["CRISPR-M word embedding"].tolist()
    crispr_m_positional_encoding_with_PAM_data = df["CRISPR-M positional encoding with PAM channel"].tolist()
    crispr_m_positional_encoding_data = df["CRISPR-M positional encoding"].tolist()

    width = 0.1
    pos1 = np.array([1,2,3,4,5,6,7,8])-width*4
    pos2 = pos1+width
    pos3 = pos2+width
    pos4 = pos3+width
    pos5 = pos4+width
    pos6 = pos5+width
    pos7 = pos6+width
    pos8 = pos7+width
    pos9 = pos8+width

    # c = ["#B97A57", "#FFC90E", "#FFF200", "#FFAEC9", "#A349A4", "#22B14C", "#00A2E8", "#ED1C24"]
    plt.bar(pos1, crispr_m_6_data, width=width, color="#B97A57", label="CRISPR-M 6 channel encoding")
    plt.bar(pos2, crispr_ip_data, width=width, color="#FFC90E", label="CRISPR-IP")
    plt.bar(pos3, crispr_ip_without_pam_data, width=width, color="#FFF200", label="CRISPR-IP without PAM channel")
    plt.bar(pos4, crispr_net_data, width=width, color="#FFAEC9", label="CRISPR-Net")
    plt.bar(pos5, crispr_net_with_pam_data, width=width, color="#A349A4", label="CRISPR-Net with PAM channel")
    plt.bar(pos6, crispr_m_word_embedding_with_PAM_data, width=width, color="#B5E61D", label="CRISPR-M word embedding with PAM channel")
    plt.bar(pos7, crispr_m_word_embedding_data, width=width, color="#22B14C", label="CRISPR-M word embedding")
    plt.bar(pos8, crispr_m_positional_encoding_with_PAM_data, width=width, color="#00A2E8", label="CRISPR-M positional encoding with PAM channel")
    plt.bar(pos9, crispr_m_positional_encoding_data, width=width, color="#ED1C24", label="CRISPR-M positional encoding")

    plt.xlim(0.4, 8.6)
    plt.ylim(0.1, 0.8)
    plt.xticks(np.arange(1,9), ["CRISPR-IP", "DNN3", "DNN5", "DNN10", "CNN3", "CNN5", "LSTM", "CRU"])
    for i in range(8):
        plt.text(pos1[i], crispr_m_6_data[i], "%.2f"%crispr_m_6_data[i], va="bottom", ha="center", rotation=90.)
        plt.text(pos2[i], crispr_ip_data[i], "%.2f"%crispr_ip_data[i], va="bottom", ha="center", rotation=90.)
        plt.text(pos3[i], crispr_ip_without_pam_data[i], "%.2f"%crispr_ip_without_pam_data[i], va="bottom", ha="center", rotation=90.)
        plt.text(pos4[i], crispr_net_data[i], "%.2f"%crispr_net_data[i], va="bottom", ha="center", rotation=90.)
        plt.text(pos5[i], crispr_net_with_pam_data[i], "%.2f"%crispr_net_with_pam_data[i], va="bottom", ha="center", rotation=90.)
        plt.text(pos6[i], crispr_m_word_embedding_with_PAM_data[i], "%.2f"%crispr_m_word_embedding_with_PAM_data[i], va="bottom", ha="center", rotation=90.)
        plt.text(pos7[i], crispr_m_word_embedding_data[i], "%.2f"%crispr_m_word_embedding_data[i], va="bottom", ha="center", rotation=90.)
        plt.text(pos8[i], crispr_m_positional_encoding_with_PAM_data[i], "%.2f"%crispr_m_positional_encoding_with_PAM_data[i], va="bottom", ha="center", rotation=90.)
        plt.text(pos9[i], crispr_m_positional_encoding_data[i], "%.2f"%crispr_m_positional_encoding_data[i], va="bottom", ha="center", rotation=90.)
    plt.legend(loc="best", ncol=2)
    plt.title('(a) Comparisons of Encoding Schemes on AUPRC')

def draw_mismatch_boxplot():
    data = dict()
    data["CFDScoring"] = [0.9894479822634211, 0.487760404999776, 0.34067071536978993, 0.3282818247144742, 0.3349452423854218, 0.3460166282006362]
    data["CNN_std"] = [0.8428844176513993, 0.013269789262044829, 0.23684210526315788, 0.02513151028812388, 0.054201696008175884, 0.023616337818172583]
    data["DeepCRISPR"] = [0.839983778856713, 0.01318384340077423, 0.2398989898989899, 0.024994114959081656, 0.0540399746116906, 0.023530424350106296]
    data["CRISPR-Net"] = [0.9886498287532989, 0.5503168790158111, 0.283816364497505, 0.25691606356656504, 0.2718744716445713, 0.280440871229953]
    data["R-CRISPR"] = [0.9876388900007176, 0.5504772140166828, 0.49773448422873073, 0.2886317032724526, 0.3841886019437356, 0.34436817774015566]
    data["CRISPR-IP"] = [0.9874117682632655, 0.20885190257845532, 0.1027906670120156, 0.10284384326109917, 0.10275257263412368, 0.10613672122729575]
    data["CRISPR-M"] = [0.9888388086999331, 0.4968462689758392, 0.4858024331929405, 0.3296979280114693, 0.40791574313927753, 0.3626395475279518]


    width = 0.1
    pos1 = np.array([1,2,3,4,5,6])-width*3
    pos2 = pos1+width
    pos3 = pos2+width
    pos4 = pos3+width
    pos5 = pos4+width
    pos6 = pos5+width
    pos7 = pos6+width

    # c = ["#B97A57", "#FFC90E", "#FFAEC9", "#A349A4", "#22B14C", "#00A2E8", "#ED1C24"]
    plt.bar(pos1, data["CFDScoring"], width=width, color="#B97A57", label="CFDScoring")
    plt.bar(pos2, data["CNN_std"], width=width, color="#FFC90E", label="CNN_std")
    plt.bar(pos3, data["DeepCRISPR"], width=width, color="#FFAEC9", label="DeepCRISPR")
    plt.bar(pos4, data["CRISPR-Net"], width=width, color="#A349A4", label="CRISPR-Net")
    plt.bar(pos5, data["R-CRISPR"], width=width, color="#22B14C", label="R-CRISPR")
    plt.bar(pos6, data["CRISPR-IP"], width=width, color="#00A2E8", label="CRISPR-IP")
    plt.bar(pos7, data["CRISPR-M"], width=width, color="#ED1C24", label="CRISPR-M")

    # plt.xlim(0.4, 6.6)
    plt.ylim(0.0, 1.1)
    plt.xticks(np.arange(1,7), ["Accuracy", "Precision", "Recall", "F1 score", "F2 score", "SRCC"])
    for i in range(6):
        plt.text(pos1[i], data["CFDScoring"][i], "%.2f"%data["CFDScoring"][i], va="bottom", ha="center", rotation=90.)
        plt.text(pos2[i], data["CNN_std"][i], "%.2f"%data["CNN_std"][i], va="bottom", ha="center", rotation=90.)
        plt.text(pos3[i], data["DeepCRISPR"][i], "%.2f"%data["DeepCRISPR"][i], va="bottom", ha="center", rotation=90.)
        plt.text(pos4[i], data["CRISPR-Net"][i], "%.2f"%data["CRISPR-Net"][i], va="bottom", ha="center", rotation=90.)
        plt.text(pos5[i], data["R-CRISPR"][i], "%.2f"%data["R-CRISPR"][i], va="bottom", ha="center", rotation=90.)
        plt.text(pos6[i], data["CRISPR-IP"][i], "%.2f"%data["CRISPR-IP"][i], va="bottom", ha="center", rotation=90.)
        plt.text(pos7[i], data["CRISPR-M"][i], "%.2f"%data["CRISPR-M"][i], va="bottom", ha="center", rotation=90.)
    plt.legend(loc="best")
    plt.title('(c) Comparisons on Mismatches-only sgRNA-Target Prediction: Performance Evaluation on Several Metrics')

def draw_sampling_boxplot():
    data = dict()
    data["Undersampling"] = [0.8876470670724012, 0.001951827242524917, 0.94, 0.003895565685868214, 0.009678747940691929, 0.979573598649729, 0.3285776411918913, 0.0400290165791733]
    data["Oversampling"] = [0.9682470680072733, 0.00644122383252818, 0.88, 0.01278883883156518, 0.031290001422272784, 0.9795437225024897, 0.28787027398711834, 0.07375165592224221]
    data["Original CIECLE"] = [0.9972584874703763, 0.5962376602526593, 0.394, 0.23207363595179192, 0.2151656038988377, 0.9616539930709781, 0.37932679616401754, 0.32647996692095604]

    width = 0.2
    pos1 = np.array([1,2,3,4,5,6,7,8])-width
    pos2 = pos1+width
    pos3 = pos2+width

    # c = ["#B97A57", "#FFC90E", "#FFAEC9", "#A349A4", "#22B14C", "#00A2E8", "#ED1C24"]
    plt.bar(pos1, data["Undersampling"], width=width, color="#22B14C", label="Undersampling")
    plt.bar(pos2, data["Oversampling"], width=width, color="#00A2E8", label="Oversampling")
    plt.bar(pos3, data["Original CIECLE"], width=width, color="#ED1C24", label="Original CIECLE")

    # plt.xlim(0.4, 6.6)
    plt.ylim(0.0, 1.1)
    plt.xticks(np.arange(1,9), ["Accuracy", "Precision", "Recall", "F1 score", "F2 score", "AUROC", "AUPRC", "SRCC"])
    for i in range(8):
        plt.text(pos1[i], data["Undersampling"][i], "%.2f"%data["Undersampling"][i], va="bottom", ha="center", rotation=90.)
        plt.text(pos2[i], data["Oversampling"][i], "%.2f"%data["Oversampling"][i], va="bottom", ha="center", rotation=90.)
        plt.text(pos3[i], data["Original CIECLE"][i], "%.2f"%data["Original CIECLE"][i], va="bottom", ha="center", rotation=90.)
    plt.legend(loc="best")
    plt.title('(i) Sampling Method Test')


if __name__ == "__main__":
    FONTSIZE = 8
    # plt.figure(dpi=300, figsize=(9, 20))
    plt.figure(dpi=300, figsize=(8, 12))
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

    ## mismatch test
    print("drawing 3mismatch roc")
    plt.subplot(3, 2, 1)
    path_list = ["CFDScoring/", "CNN_std/", "DeepCRISPR/", "CRISPR-Net/", "R-CRISPR/", "CRISPR-IP/", "mine/"] #CFD pickle is v2.7, cannot use in here v3
    c = ["#B97A57", "#FFC90E", "#FFAEC9", "#A349A4", "#22B14C", "#00A2E8", "#ED1C24"]
    model_name_list = ["CFDScoring", "CNN_std", "DeepCRISPR", "CRISPR-Net", "R-CRISPR", "CRISPR-IP", "CRISRP-M"]
    for i in range(7):
        print("drawing"+model_name_list[i])
        draw_mean_roc(PATH+"/3mismatch/"+path_list[i], c[i], model_name_list[i])
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='grey', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('(a) Comparisons on Mismatches-only sgRNA-Target Prediction:\nMean Receiver Operating Characteristic Curve')
    plt.legend(loc="best")

    print("drawing 3mismatch prc")
    plt.subplot(3, 2, 2)
    for i in range(7):
        draw_mean_prc(PATH+"/3mismatch/"+path_list[i], c[i], model_name_list[i])
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('(b) Comparisons on Mismatches-only sgRNA-Target Prediction:\nMean Precision-Recall Curve')
    plt.legend(loc="best")

    print("drawing 3mismatch boxplot")
    plt.subplot(3, 2, (3,4))
    draw_mismatch_boxplot()

    ## 4multidata test
    print("drawing 4multidata boxplot")
    plt.subplot(3, 2, 5)
    path_list = ["CRISPR-Net/", "R-CRISPR/", "CRISPR-IP/", "CRISPR-M/"] 
    c = ["#FFC90E", "#22B14C", "#00A2E8", "#ED1C24"]
    model_name_list = ["CRISPR-Net", "R-CRISPR", "CRISPR-IP", "CRISRP-M"]
    for i in range(4):
        draw_mean_roc(PATH+"/4multidataset/"+path_list[i], c[i], model_name_list[i])
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='grey', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('(d) Comparisons with Complex Off-Target Site Datasets:\nMean Receiver Operating Characteristic Curve')
    plt.legend(loc="best")

    plt.subplot(3, 2, 6)
    for i in range(4):
        draw_mean_prc(PATH+"/4multidataset/"+path_list[i], c[i], model_name_list[i])
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('(e) Comparisons with Complex Off-Target Site Datasets:\nMean Precision-Recall Curve')
    plt.legend(loc="best")

    plt.savefig(fname="fig2.1.svg", format="svg", bbox_inches="tight")
    plt.savefig(fname="fig2.1.tif", format="tif", bbox_inches="tight")
    plt.savefig(fname="fig2.1.png", format="png", bbox_inches="tight")

    plt.figure(dpi=300, figsize=(8, 8))
    ## encoding test
    print("drawing 2encoding")
    plt.subplot(2, 2, (1,2))
    draw_encoding_test()

    ## epigenetic test
    print("drawing epigenetic")
    plt.subplot(2, 2, 3)
    path_list = ["DeepCRISPR/", "CRISPR-M/without_epigenetic/", "CRISPR-M/with_epigenetic/"]
    c = ["#22B14C", "#00A2E8", "#ED1C24"]
    model_name_list = ["DeepCRISPR", "CRISPR-M without epigenetic features", "CRISPR-M with epigenetic features"]
    for i in range(3):
        draw_mean_roc(PATH+"/6epigenetic/"+path_list[i], c[i], model_name_list[i])
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='grey', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('(b) Comparisons with Epigenetic Features:\nMean Receiver Operating Characteristic Curve')
    plt.legend(loc="best", prop={"size":6})

    plt.subplot(2, 2, 4)
    for i in range(3):
        draw_mean_prc(PATH+"/6epigenetic/"+path_list[i], c[i], model_name_list[i])
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('(c) Comparisons with Epigenetic Features:\nMean Precision-Recall Curve')
    plt.legend(loc="best", prop={"size":6})

    plt.savefig(fname="fig2.2.svg", format="svg", bbox_inches="tight")
    plt.savefig(fname="fig2.2.tif", format="tif", bbox_inches="tight")
    plt.savefig(fname="fig2.2.png", format="png", bbox_inches="tight")

    ## sampling test
    # print("drawing sampling")
    # plt.subplot(6, 2, (11,12))
    # draw_sampling_boxplot()
    # plt.subplot(6, 2, 11)
    # path_list = ["undersampling/", "oversampling/", "2CIRCLE_GUIDE/"]
    # c = ["#22B14C", "#00A2E8", "#ED1C24"]
    # model_name_list = ["Undersampling", "Oversampling", "Origin"]
    # for i in range(3):
    #     draw_mean_roc(PATH+"/8sampling/"+path_list[i], c[i], model_name_list[i])
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='grey', alpha=.8)
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate (FPR)')
    # plt.ylabel('True Positive Rate (TPR)')
    # plt.title('(i) Sampling Test: Mean Receiver Operating Characteristic Curve')
    # plt.legend(loc="best")

    # plt.subplot(6, 2, 12)
    # for i in range(3):
    #     draw_mean_prc(PATH+"/8sampling/"+path_list[i], c[i], model_name_list[i])
    # plt.plot([0, 1], [1, 0], linestyle='--', lw=1, color='grey', alpha=.8)
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.ylabel('Precision')
    # plt.xlabel('Recall')
    # plt.title('(j) Sampling Test: Mean Precision-Recall Curve')
    # plt.legend(loc="best")

    # plt.savefig(fname="fig2.svg", format="svg", bbox_inches="tight")
    # plt.savefig(fname="fig2.png", format="png", bbox_inches="tight")





#################################################3
    # x = [2, 6, 7, 8, 10, 20]
    # metrics = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC"]
    # models = ["DNN3", "CNN3", "LSTM", "CRISPR-IP"]
    # c = ["#B97A57", "#FFC90E", "#FFF200", "#FFAEC9", "#A349A4", "#22B14C", "#00A2E8", "#ED1C24"]

    # df = pd.read_excel("DNN3.xlsx", sheet_name="Accurary")
    # l = df.iloc[1].tolist()
    # print(l)

    # for k in range(4):
    #     for j in range(6):
    #         plt.subplot(4, 6, 1+j+6*k)
    #         df = pd.read_excel(models[k]+'.xlsx', sheet_name=metrics[j])
    #         for i in range(8):
    #             l = df.iloc[i].tolist()
    #             plt.plot(x, l[1:], label=l[0], color=c[i])
    #         plt.xticks(x)
    #         plt.title(models[k]+"-"+metrics[j])
    # plt.legend(loc="best")
    # plt.savefig(fname="1indel.svg", format="svg", bbox_inches="tight")