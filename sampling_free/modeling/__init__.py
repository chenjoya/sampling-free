from .generalized_rcnn import GeneralizedRCNN

MODELS = {"GeneralizedRCNN": GeneralizedRCNN}

def build_model(cfg):
    model = MODELS[cfg.MODEL.ARCHITECTURE]
    return model(cfg)