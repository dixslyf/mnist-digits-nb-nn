def check_model(model):
    if model not in ("nn", "nb"):
        raise ValueError(f"Invalid model {model}")
