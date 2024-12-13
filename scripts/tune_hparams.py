# scripts/tune_hparams.py
import hydra
from omegaconf import DictConfig
import optuna
import mlflow
import mlflow.pytorch
from episteme.train import run_training

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    def objective(trial):
        # Suggest new hyperparams
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        cfg.model.lr = lr

        # Start a new nested MLflow run for this trial
        with mlflow.start_run(nested=True):
            # Enable autologging inside this run
            mlflow.pytorch.autolog(log_models=False)

            # Log the parameters for this trial
            mlflow.log_param("lr", lr)

            # Run training for this trial
            run_training(cfg)

            # Return a dummy metric (0.0) or ideally fetch a real metric
            return 0.0

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    print("Best hyperparameters:", study.best_params)

if __name__ == "__main__":
    main()
