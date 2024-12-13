import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from .data import KFoldDataModule
from .model import MLP

def set_seed(seed=42):
    pl.seed_everything(seed, workers=True)

def run_training(cfg):
    set_seed(cfg.data.seed)

    torch.set_float32_matmul_precision('high')

    dm = KFoldDataModule(
        num_samples=cfg.data.num_samples,
        num_features=cfg.data.num_features,
        test_size=cfg.data.test_size,
        k_folds=cfg.data.k_folds,
        current_fold=cfg.data.current_fold,
        seed=cfg.data.seed,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers
    )
    dm.setup()

    model = MLP(
        input_dim=cfg.model.input_dim,
        hidden_layers=cfg.model.hidden_layers,
        output_dim=cfg.model.output_dim,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        dropout=cfg.model.dropout
    )

    early_stop = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=5,
        verbose=True
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        verbose=True,
        filename="best_model-epoch={epoch:02d}-acc={val_acc:.4f}"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        callbacks=[early_stop, checkpoint_callback]
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
