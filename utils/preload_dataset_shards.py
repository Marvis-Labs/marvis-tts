import argparse
from datasets import load_dataset
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the training configuration file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_data = json.load(f)

    _ = load_dataset(config_data["dataset_repo_id"])
