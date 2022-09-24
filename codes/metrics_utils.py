import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, auc, precision_score, recall_score, f1_score, fbeta_score, accuracy_score

def compute_auroc_and_auprc(model, out_dim, test_features, test_labels):
    if out_dim == 1:
        y_pred = model.predict(x=test_features).ravel()
        pred_labels = list()
        # if is_label_binarized: # is_label_binarized True==classification False==regression
        for i in y_pred:
            if i >= 0.5:
                pred_labels.append(1.0)
            else:
                pred_labels.append(0.0)
        pred_labels = np.array(pred_labels)
        pred_score = y_pred
        validation_labels = np.array(test_labels, dtype=np.float32)
    elif out_dim == 2:
        y_pred = model.predict(x=test_features)
        pred_labels = np.argmax(y_pred, axis=1)
        pred_score = y_pred[:, 1]
        validation_labels = test_labels[:, 1]


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
    return accuracy, precision, recall, f1, fbeta, auroc, auprc, auroc_by_auc, auprc_by_auc, spearman_corr_by_pred_score, spearman_corr_by_pred_labels, fpr, tpr, precision_point, recall_point

