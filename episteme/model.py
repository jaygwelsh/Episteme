import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, AUROC
import os

class SimpleBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class MLP(pl.LightningModule):
    def __init__(self, input_dim, hidden_layers=[4096,4096,4096,4096], output_dim=1, lr=0.001, weight_decay=1e-5, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        in_dim = self.hparams.input_dim
        for h in self.hparams.hidden_layers:
            layers.append(SimpleBlock(in_dim, h, dropout=self.hparams.dropout))
            in_dim = h
        self.layers = nn.ModuleList(layers)
        self.final_linear = nn.Linear(in_dim, self.hparams.output_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.val_acc = Accuracy(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_auc = AUROC(task="binary")

        self.val_logits = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).view(-1)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self(x).view(-1)
        self.val_logits.append(y_hat.detach().cpu())
        self.val_labels.append(y.detach().cpu())
        return {}

    def on_validation_epoch_end(self):
        if len(self.val_logits)==0:
            return
        all_logits = torch.cat(self.val_logits)
        all_labels = torch.cat(self.val_labels)
        self.val_logits.clear()
        self.val_labels.clear()

        # Just compute metrics without calibration here
        preds = torch.sigmoid(all_logits)
        acc = self.val_acc((preds>=0.5).float(), all_labels)
        f1 = self.val_f1((preds>=0.5).float(), all_labels)
        precision = self.val_precision((preds>=0.5).float(), all_labels)
        recall = self.val_recall((preds>=0.5).float(), all_labels)
        auc = self.val_auc(preds, all_labels)
        val_loss = self.loss_fn(all_logits, all_labels)

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_auc", auc, prog_bar=True)

    def test_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x).view(-1)
        preds = torch.sigmoid(y_hat)
        self.test_preds.append(preds.detach().cpu())
        self.test_labels.append(y.detach().cpu())
        return {}

    def on_test_epoch_end(self):
        if len(self.test_preds)==0:
            return
        all_preds = torch.cat(self.test_preds)
        all_labels = torch.cat(self.test_labels)
        # Save predictions for ensembling
        fold = self.trainer.datamodule.current_fold if hasattr(self.trainer.datamodule,'current_fold') else 0
        torch.save(all_preds, f"fold_{fold}_preds.pt")
        torch.save(all_labels, f"fold_{fold}_labels.pt")

        # Compute final metrics here (though not final if we do ensembling)
        acc = self.val_acc((all_preds>=0.5).float(), all_labels)
        f1 = self.val_f1((all_preds>=0.5).float(), all_labels)
        precision = self.val_precision((all_preds>=0.5).float(), all_labels)
        recall = self.val_recall((all_preds>=0.5).float(), all_labels)
        auc = self.val_auc(all_preds, all_labels)

        # Compute test_loss
        # Need logits for stable loss
        # If we want stable test_loss with raw logits, we must store test logits too
        # For simplicity, let's store them:
        # Let's store logits also and compute test_loss:
        # But we didn't store logits in test_step. Let's do that:
        # We'll store test_logits similarly to test_preds:
        # Actually we didn't store them above. Let's fix that. We'll store raw logits too.

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        steps_per_epoch = self.trainer.estimated_stepping_batches
        total_steps = steps_per_epoch*self.trainer.max_epochs
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.hparams.lr, total_steps=total_steps, pct_start=0.1
            ),
            "interval":"step",
            "frequency":1
        }
        return [optimizer],[scheduler]
