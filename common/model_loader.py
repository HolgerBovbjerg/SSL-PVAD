import torch.nn

from models.PVAD1_et import PVAD1ET, PVAD1ET2, PVAD1ET22, PVAD1ET3, PVAD1ET4
from models.PVAD1_sc import PVAD1_SC
from models.modules.LSTMEncoder import LSTMEncoder, LSTMEncoder2


def get_model(model_config, **kwargs):
    model_classes = {
        "PVAD1_ET": PVAD1ET,
        "PVAD1_ET2": PVAD1ET2,
        "PVAD1_ET22": PVAD1ET22,
        "PVAD1_ET3": PVAD1ET3,
        "PVAD1_ET4": PVAD1ET4,
        "PVAD1_SC": PVAD1_SC,
        "LSTMEncoder": LSTMEncoder,
        "LSTMEncoder2": LSTMEncoder2
    }

    model_name = model_config["name"]
    if model_name in model_classes:
        return model_classes[model_name](**model_config["settings"], **kwargs)
    else:
        raise ValueError(f"Model name {model_name} not recognized")


def load_pretrained_model(model: torch.nn.Module, checkpoint_path: str = "", map_location: str = "cpu"):
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_state_dict = model.state_dict()
    checkpoint_model_state_dict = checkpoint["model_state_dict"]
    if "lstm.weight_ih_l0" in checkpoint_model_state_dict:
        checkpoint_model_state_dict = {key.replace("lstm.", ""): value for key, value in checkpoint_model_state_dict.items()}

    print(model_state_dict.keys())
    print(checkpoint_model_state_dict.keys())
    model_state_dict.update(checkpoint_model_state_dict)
    model.load_state_dict(model_state_dict)
    print(f"Loaded checkpoint {checkpoint_path}.")
    return model


def load_pretrained_encoder(model: torch.nn.Module, checkpoint_path: str = "", map_location: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_state_dict = model.encoder.state_dict()
    checkpoint_model_state_dict = checkpoint["model_state_dict"]
    model_state_dict.update(checkpoint_model_state_dict)

    model.encoder.load_state_dict(model_state_dict)
    print(f"Loaded checkpoint {checkpoint_path}.")
    return model

def load_pretrained_LSTM2_to_LSTM1(model: torch.nn.Module, checkpoint_path: str = "", map_location: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_state_dict = model.state_dict()
    checkpoint_model_state_dict = checkpoint["model_state_dict"]
    # model_state_dict.update(checkpoint_model_state_dict)
    model_state_dict["lstm.weight_ih_l0"] = checkpoint_model_state_dict["rnns.0.weight_ih_l0"]
    model_state_dict["lstm.bias_ih_l0"] = checkpoint_model_state_dict["rnns.0.bias_ih_l0"]
    model_state_dict["lstm.weight_hh_l0"] = checkpoint_model_state_dict["rnns.0.weight_hh_l0"]
    model_state_dict["lstm.bias_hh_l0"] = checkpoint_model_state_dict["rnns.0.bias_hh_l0"]
    model_state_dict["lstm.weight_ih_l1"] = checkpoint_model_state_dict["rnns.1.weight_ih_l0"]
    model_state_dict["lstm.bias_ih_l1"] = checkpoint_model_state_dict["rnns.1.bias_ih_l0"]
    model_state_dict["lstm.weight_hh_l1"] = checkpoint_model_state_dict["rnns.1.weight_hh_l0"]
    model_state_dict["lstm.bias_hh_l1"] = checkpoint_model_state_dict["rnns.1.bias_hh_l0"]

    model.load_state_dict(model_state_dict)
    print(f"Loaded checkpoint {checkpoint_path}.")
    return model
