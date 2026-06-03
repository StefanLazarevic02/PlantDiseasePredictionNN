import yaml
from src.preprocessing.dataset_analyzer import DatasetAnalyzer

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

analyzer = DatasetAnalyzer(
    root_dir=config["dataset"]["root_dir"],
    train_dir=config["dataset"]["train_dir"],
    valid_dir=config["dataset"]["valid_dir"],
)

# Prikazuju se statistiku
analyzer.print_summary()

# Iscrtavanje grafika distribucije klasa u datasetu
analyzer.plot_distribution(save_path="outputs/class_distribution.png")