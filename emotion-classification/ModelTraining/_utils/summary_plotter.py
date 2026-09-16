from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, PrecisionRecallDisplay, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def roc_auc_curves(mode, models):
    for model in models:
        Y_bin = label_binarize(model['y'], classes=sorted(model['y'].unique()))
        n_classes = Y_bin.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        lw = 2
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(Y_bin[:, i], model['proba'].iloc[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # micro average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        roc_auc_mean = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, lw=2,
                 label=f"Micro-average ROC curve for {model['name']} (AUC = {roc_auc_mean:0.2f})")



    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{mode} ROC for test')
    plt.legend(loc="lower right")
    plt.savefig(f'../Results/img/summary_plots/test2.png')
    plt.clf()


def precision_recall_curves(models, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    # Iterate over each model and plot the micro-averaged precision-recall curve
    for model in models:
        Y_bin = label_binarize(model['y'], classes=sorted(model['y'].unique()))
        n_classes = Y_bin.shape[1]

        precision = dict()
        recall = dict()

        for i in range(n_classes):
            prec, rec, _ = precision_recall_curve(Y_bin[:, i], model['proba'].iloc[:, i])
            precision[i] = prec
            recall[i] = rec
        y_true_flat = Y_bin.ravel()
        y_score_flat = model['proba'].values.ravel()
        prec_micro, rec_micro, _ = precision_recall_curve(y_true_flat, y_score_flat)

        # Plot the micro-averaged precision-recall curve for the current model
        display = PrecisionRecallDisplay(
            recall=rec_micro,
            precision=prec_micro,
            average_precision=average_precision_score(y_true_flat, y_score_flat, average="micro"),
            prevalence_pos_label=Counter(y_true_flat)[1] / len(y_true_flat),
        )

        display.plot(ax=ax, label=f'P-R curve for {model["name"]}') #
        lines = ax.lines

    f_scores = np.linspace(0.2, 0.8, num=4)

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    _ = ax.set_title(f"Micro-averaged Precision-Recall Curve over {title}")
    _ = ax.legend()

    plt.savefig(f'../Results/img/summary_plots/{title}.png')
    plt.clf()
    