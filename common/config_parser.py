import yaml
import os
import torch


def get_config(config_file: str) -> dict:
    """Reads settings from config file.
    Args:
        config_file (str): YAML config file.
    Returns:
        dict: Dict containing settings.
    """

    with open(config_file, "r", encoding="utf8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if config["exp"]["wandb"] and config["exp"]["wandb_api_key"] is not None:
        assert os.path.exists(config["exp"]["wandb_api_key"]), f"API key file not found at specified location " \
                                                           f"{config['exp']['wandb_api_key']}."

    if config["exp"]["device"] == "auto":
        if torch.cuda.is_available():
            config["exp"]["device"] = torch.device("cuda")
        elif torch.torch.backends.mps.is_available():
            config["exp"]["device"] = torch.device("mps")
        else:
            config["exp"]["device"] = torch.device("cpu")
        print(f"Using device: {config['exp']['device']}")
    return config
