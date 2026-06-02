import numpy
import cv2
import sklearn.metrics as metrics

def get_roc_auc(trues, preds):
    labels = [0,1,2,3]
    nb_classes = len(labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    print(trues, preds)
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(trues[:, i], preds[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(trues.ravel(), preds.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = numpy.unique(numpy.concatenate([fpr[i] for i in range(nb_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = numpy.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += numpy.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= nb_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
    print('roc_auc = ', roc_auc["macro"])