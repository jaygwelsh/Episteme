# tests/test_training.py
import pytest
from omegaconf import OmegaConf
from episteme.train import run_training

@pytest.fixture
def test_config():
    # Minimal configuration override for testing
    cfg = OmegaConf.create({
        "trainer": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "enable_checkpointing": False,
            "log_every_n_steps": 1
        },
        "model": {
            "input_dim": 20,
            "hidden_dim": 16,
            "output_dim": 1,
            "lr": 0.001
        },
        "data": {
            "num_samples": 100,
            "input_dim": 20,
            "batch_size": 16,
            "seed": 42
        },
        "logging": {
            "mlflow_experiment_name": "Test_Experiments",
            "mlflow_tracking_uri": "file:./test_mlruns"
        }
    })
    return cfg

def test_training_runs_without_error(test_config):
    # If run_training completes without exception, it's a success.
    # Pytest will fail if an exception is raised.
    run_training(test_config)
