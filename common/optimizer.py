from torch import nn, optim


def get_optimizer(net: nn.Module, opt_config: dict) -> optim.Optimizer:
    """Creates optimizer based on config.
    Args:
        net (nn.Module): Model instance.
        opt_config (dict): Dict containing optimizer settings.
    Raises:
        ValueError: Unsupported optimizer type.
    Returns:
        optim.Optimizer: Optimizer instance.
    """

    if isinstance(net, list):
        parameters = []
        for model in net:
            parameters += list(filter(lambda p: p.requires_grad, model.parameters()))
    else:
        parameters = list(filter(lambda p: p.requires_grad, net.parameters()))

    if opt_config["opt_type"] == "Adam":
        optimizer = optim.Adam(parameters, **opt_config["opt_kwargs"])
    elif opt_config["opt_type"] == "AdamW":
        optimizer = optim.AdamW(parameters, **opt_config["opt_kwargs"])
    elif opt_config["opt_type"] == "SGD":
        optimizer = optim.SGD(parameters, **opt_config["opt_kwargs"])
    else:
        raise ValueError(f'Unsupported optimizer {opt_config["opt_type"]}')

    return optimizer
