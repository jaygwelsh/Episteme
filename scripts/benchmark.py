import hydra
from omegaconf import DictConfig
from episteme.train import run_training
import os

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # We'll run training & testing for each fold and save predictions
    # Adjust current_fold in cfg and run sequentially
    # After each fold run, predictions are saved by model code (we'll modify model.py)
    # Once all folds are done, we have predictions from each fold.

    k_folds = cfg.data.k_folds
    original_fold = cfg.data.current_fold
    out_dir = os.getcwd()

    for fold in range(k_folds):
        cfg.data.current_fold = fold
        print(f"Running fold {fold}/{k_folds-1}...")
        run_training(cfg)
        # Predictions and labels saved by model

    # After running all folds, run ensemble.py manually:
    print("All folds completed. Run `python scripts/ensemble.py` to ensemble predictions.")

if __name__ == "__main__":
    main()
