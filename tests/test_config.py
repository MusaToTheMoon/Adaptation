from scripts.utils import load_config

cfg = load_config("configs/task1/mistral_small.yaml")

print(cfg['model']['offline'])