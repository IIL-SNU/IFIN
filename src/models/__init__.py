from .ifin import (
    FSO,
    IFIB,
    IFINNet,
    IFIN,
    ISO,
    RB,
    build_ifin_model,
)

__all__ = [
    "IFINNet",
    "IFIB",
    "ISO",
    "FSO",
    "RB",
    "build_ifin_model",
    "IFIN",
    "build_model",
]


def build_model(config, psf, device="cpu", checkpoint=None):
    name = config["model"].get("name", "ifin").lower()
    if name == "ifin":
        model = build_ifin_model(config, psf, checkpoint=checkpoint)
    else:
        from .baselines import build_baseline
        model = build_baseline(config, psf, device)
    return model.to(device)
