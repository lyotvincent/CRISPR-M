#Calculates the Cutting Frequency Determination score
#Requirements: 1. Pickle file with mismatch scores in working directory
#              2. Pickle file containing PAM scores in working directory 
#Input: 1. 23mer WT sgRNA sequence
#       2. 23mer Off-target sgRNA sequence
#Output: CFD score
import pickle, random, os
import pandas as pd
import argparse
import re, time, sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, auc, precision_score, recall_score, f1_score, fbeta_score, accuracy_score

sys.path.append("../../codes")
from load_4_cfd import load_PKD, load_PDH, load_SITE, load_GUIDE_I, load_GUIDE_II, load_GUIDE_III

def get_parser():
    parser = argparse.ArgumentParser(description='Calculates CFD score')
    parser.add_argument('--wt',
        type=str,
        help='WT 23mer sgRNA sequence')
    parser.add_argument('--off',
        type=str,
        help='Off-target 23mer sgRNA sequence')
    return parser

#Reverse complements a given string
def revcom(s):
    basecomp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A','U':'A'}
    letters = list(s[::-1])
    letters = [basecomp[base] for base in letters]
    return ''.join(letters)

#Unpickle mismatch scores and PAM scores
def get_mm_pam_scores():
    try:
        mm_scores = pickle.load(open('mismatch_score.pkl','rb'))
        pam_scores = pickle.load(open('pam_scores.pkl','rb'))
        return (mm_scores,pam_scores)
    except: 
        raise Exception("Could not find file with mismatch scores or PAM scores")

#Calculates CFD score
def calc_cfd(wt,sg,pam):
    mm_scores,pam_scores = get_mm_pam_scores()
    score = 1
    sg = sg.replace('T','U')
    wt = wt.replace('T','U')
    # print "sg:", sg
    # print "DNA", wt
    s_list = list(sg)
    wt_list = list(wt)
    for i,sl in enumerate(s_list):
        if wt_list[i] == sl:
            score*=1
        else:
            key = 'r'+wt_list[i]+':d'+revcom(sl)+','+str(i+1)
            # print key
            score*= mm_scores[key]
    score*=pam_scores[pam]
    return (score)

def test(x_on, x_off, y):
    y_pred = list()
    for i in range(len(x_on)):
        if i % 10000 == 0:
            print (i)
        wt =  x_on[i]
        off = x_off[i]
        m_wt = re.search('[^ATCG]',wt)
        # print "1", m_wt
        m_off = re.search('[^ATCG]',off)
        # print "2", m_off
        if (m_wt is None) and (m_off is None):
            pam = off[-2:]
            sg = off[:-3]
            # print wt, off, sg, pam
            cfd_score = calc_cfd(wt,sg,pam)
            # print "CFD score: "+str(cfd_score)
            y_pred.append(cfd_score)

    pred_labels = list()
    for i in y_pred:
        if i >= 0.5:
            pred_labels.append(1.0)
        else:
            pred_labels.append(0.0)
    pred_labels = np.array(pred_labels)
    pred_score = y_pred
    validation_labels = y

    accuracy = accuracy_score(validation_labels, pred_labels)
    precision = precision_score(validation_labels, pred_labels)
    recall = recall_score(validation_labels, pred_labels)
    f1 = f1_score(validation_labels, pred_labels)
    fbeta = fbeta_score(y_true=validation_labels, y_pred=pred_labels, beta=2)

    auroc = roc_auc_score(validation_labels, pred_score)
    fpr, tpr, thresholds = roc_curve(validation_labels, pred_score)
    auroc_by_auc = auc(fpr, tpr)

    auprc = average_precision_score(validation_labels, pred_score)
    precision_point, recall_point, thresholds = precision_recall_curve(validation_labels, pred_score)
    precision_point[(recall_point==0)] = 1.0
    auprc_by_auc = auc(recall_point, precision_point)

    # Spearman's rank correlation coefficient
    df = pd.DataFrame({"y_pred": pred_score, "y_label": validation_labels})
    spearman_corr_by_pred_score = df.corr("spearman")["y_pred"]["y_label"]
    df = pd.DataFrame({"y_pred": pred_labels, "y_label": validation_labels})
    spearman_corr_by_pred_labels = df.corr("spearman")["y_pred"]["y_label"]

    print("accuracy=%s, precision=%s, recall=%s, f1=%s, fbeta=%s auroc=%s, auprc=%s, auroc_by_auc=%s, auprc_by_auc=%s, spearman_corr_by_pred_score=%s, spearman_corr_by_pred_labels=%s"%(accuracy, precision, recall, f1, fbeta, auroc, auprc, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels))
    # fpr_list, tpr_list, precision_point_list, recall_point_list = list(), list(), list(), list()
    # fpr_list.append(fpr)
    # tpr_list.append(tpr)
    # precision_point_list.append(precision_point)
    # recall_point_list.append(recall_point)
    # with open("fpr.csv", "wb") as f:
    #     pickle.dump(fpr_list, f)
    # with open("tpr.csv", "wb") as f:
    #     pickle.dump(tpr_list, f)
    # with open("precision_point.csv", "wb") as f:
    #     pickle.dump(precision_point_list, f)
    # with open("recall_point.csv", "wb") as f:
    #     pickle.dump(recall_point_list, f)
    
    return accuracy, precision, recall, f1, fbeta, auroc, auprc, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point


if __name__ == '__main__':

    time1 = time.time()

    SEED = int(sys.argv[1])
    print(f"SEED={SEED}")
    random.seed(SEED)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(SEED)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(SEED)  # numpy的随机性
    tf.random.set_seed(SEED)  # tensorflow的随机性

    six_dataset_fold = dict()
    six_dataset_fold["pkd"] = {"on_features": list(), "off_features": list(), "labels": list()}
    six_dataset_fold["pdh"] = {"on_features": list(), "off_features": list(), "labels": list()}
    six_dataset_fold["site"] = {"on_features": list(), "off_features": list(), "labels": list()}
    six_dataset_fold["guide_i"] = {"on_features": list(), "off_features": list(), "labels": list()}
    six_dataset_fold["guide_ii"] = {"on_features": list(), "off_features": list(), "labels": list()}
    six_dataset_fold["guide_iii"] = {"on_features": list(), "off_features": list(), "labels": list()}
    classification_data_abbr = ["site", "pkd",  "guide_ii", "guide_iii"]
    classification_data_method = [load_SITE, load_PKD, load_GUIDE_II, load_GUIDE_III]
    regression_data_abbr = ["pdh", "guide_i"]
    regression_data_method = [load_PDH, load_GUIDE_I]
    for i in range(4):
        result = classification_data_method[i]()
        six_dataset_fold[classification_data_abbr[i]]["on_features"] = result[0]
        six_dataset_fold[classification_data_abbr[i]]["off_features"] = result[1]
        six_dataset_fold[classification_data_abbr[i]]["labels"] = result[2]
    test_loss_sum, test_acc_sum, auroc_sum, auprc_sum, accuracy_sum, precision_sum, recall_sum, f1_sum, fbeta_sum, auroc_skl_sum, auprc_skl_sum, auroc_by_auc_sum, auprc_by_auc_sum, spearman_corr_by_pred_score_sum, spearman_corr_by_pred_labels_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    # fpr_list, tpr_list, precision_point_list, recall_point_list = list(), list(), list(), list()
    for i in range(2):
        x_on = list()
        x_off = list()
        y = list()
        for j in range(4):
            if bool(j) == bool(i):
                x_on.extend(six_dataset_fold[classification_data_abbr[j]]["on_features"])
                x_off.extend(six_dataset_fold[classification_data_abbr[j]]["off_features"])
                y.extend(six_dataset_fold[classification_data_abbr[j]]["labels"])
        x_on = np.array(x_on)
        x_off = np.array(x_off)
        y = np.array(y, dtype=np.float32)
        print("[INFO] Encoded dataset x_on with size of", x_on.shape)
        print("[INFO] Encoded dataset x_off with size of", x_off.shape)
        print("[INFO] The labels number of active off-target sites in dataset ytrain is {0}, the active+inactive is {1}.".format(len(y[y>0]), len(y)))

        print('Training!!')
        result = test(x_on, x_off, y)
        test_loss_sum += 0
        test_acc_sum += 0
        auroc_sum += 0
        auprc_sum += 0
        accuracy_sum += result[0]
        precision_sum += result[1]
        recall_sum += result[2]
        f1_sum += result[3]
        fbeta_sum += result[4]
        auroc_skl_sum += result[5]
        auprc_skl_sum += result[6]
        auroc_by_auc_sum += result[7]
        auprc_by_auc_sum += result[8]
        spearman_corr_by_pred_score_sum += result[9]
        spearman_corr_by_pred_labels_sum += result[10]
    print('End of the training!!')
    print("average_auroc_by_auc=%s, average_auprc_by_auc=%s" % (auroc_by_auc_sum/2, auprc_by_auc_sum/2))
    f = open("cfd-score_mismatch_test_result.txt", "a")
    f.write("seed=%s, average_auroc_by_auc=%s, average_auprc_by_auc=%s\n" % (SEED, auroc_by_auc_sum/2, auprc_by_auc_sum/2))
    f.close()
    print(time.time()-time1)



# seed=0, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=10, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=20, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=30, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=40, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=50, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=60, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=70, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=80, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=90, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=100, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=110, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=120, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=130, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=140, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=150, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=160, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=170, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=180, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238
# seed=190, average_auroc_by_auc=0.8473563780908389, average_auprc_by_auc=0.3325722401377238








    # args = get_parser().parse_args()
    # mm_scores,pam_scores = get_mm_pam_scores()
    # wt = args.wt
    # off = args.off
    # wt =  "AGTCTGAGTCGGAGCCAGGGGGG"
    # off = "GGTCTGAGTCGGAGCCAGGGCGG"
    # m_wt = re.search('[^ATCG]',wt)
    # print "1", m_wt
    # m_off = re.search('[^ATCG]',off)
    # print "2", m_off
    # if (m_wt is None) and (m_off is None):
    #     pam = off[-2:]
    #     sg = off[:-3]
    #     print wt, off, sg, pam
    #     cfd_score = calc_cfd(wt,sg,pam)
    #     print "CFD score: "+str(cfd_score)


