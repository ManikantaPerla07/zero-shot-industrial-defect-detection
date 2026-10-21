from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    """Load the project configuration from YAML."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    config = load_config()

    print("Configuration loaded successfully.")
    print(f"Project: {config['project']['name']}")
    print(f"Version: {config['project']['version']}")
    print(f"Image size: {config['image']['size']}")
