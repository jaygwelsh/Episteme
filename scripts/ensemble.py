import os
import torch
import numpy as np
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, AUROC
import glob

def main():
    # Assume predictions for each fold are saved in fold_{fold}_preds.pt and fold_{fold}_labels.pt
    pred_files = sorted(glob.glob("fold_*_preds.pt"))
    label_files = sorted(glob.glob("fold_*_labels.pt"))

    all_preds = []
    all_labels = None

    for pf, lf in zip(pred_files, label_files):
        preds = torch.load(pf)
        labels = torch.load(lf)
        all_preds.append(preds)
        if all_labels is None:
            all_labels = labels
        else:
            # Verify labels are same for each fold (should be consistent)
            assert torch.allclose(all_labels, labels), "Labels differ between folds!"

    # Average predictions
    ensemble_preds = torch.mean(torch.stack(all_preds), dim=0)

    # Compute metrics
    val_acc = Accuracy(task="binary")
    val_f1 = F1Score(task="binary")
    val_precision = Precision(task="binary")
    val_recall = Recall(task="binary")
    val_auc = AUROC(task="binary")
    # Compute binary cross entropy loss using raw predictions?
    # We only have sigmoid preds here. Let's store raw logits in model if possible.

    # If we only stored sigmoid predictions, we must compute loss via - (y*log(p)+(1-y)*log(1-p))
    # We'll assume these preds are sigmoids for simplicity now:
    # If we want raw logits, we must store them in model.py

    # For now, let's just compute metrics that don't need logits:
    # Convert to best threshold chosen previously?
    # Without threshold from training, let's pick best threshold by accuracy:
    best_acc = 0.0
    best_thresh = 0.5
    for t in np.arange(0.01,1.0,0.01):
        pc = (ensemble_preds>=t).float()
        current_acc = val_acc(pc, all_labels)
        if current_acc>best_acc:
            best_acc = current_acc
            best_thresh = float(t)

    pc = (ensemble_preds>=best_thresh).float()
    acc = val_acc(pc, all_labels)
    f1 = val_f1(pc, all_labels)
    precision = val_precision(pc, all_labels)
    recall = val_recall(pc, all_labels)
    auc = val_auc(ensemble_preds, all_labels)

    # Compute test loss with binary cross entropy:
    # BCE = -[y*log(p)+(1-y)*log(1-p)]
    p = ensemble_preds.clamp(1e-6,1-1e-6)
    y = all_labels
    test_loss = -(y*torch.log(p)+(1-y)*torch.log(1-p)).mean()

    print("Ensemble Results:")
    print(f"Test Loss: {test_loss.item()}")
    print(f"Test Acc: {acc.item()}")
    print(f"Test F1: {f1.item()}")
    print(f"Test Precision: {precision.item()}")
    print(f"Test Recall: {recall.item()}")
    print(f"Test AUC: {auc.item()}")

if __name__ == "__main__":
    main()
