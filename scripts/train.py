# scripts/train.py
import hydra
from omegaconf import DictConfig
from Episteme.train import run_training

@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: DictConfig):
    run_training(cfg)

if __name__ == "__main__":
    main()
