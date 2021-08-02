
from importlib import import_module
import sys
sys.path.append("../")

def get_anchorfree_model(model_name,config):

    #from detectorv2.model.anchor_free
    #model = getattr(backbone,model_name)()

    model = import_module("model.anchor_free." + model_name)
    model = model.get_model()
    print("model:",model_name)
    return model

def get_anchorbased_model(model_name,config):
    model = import_module("model.anchor_based." + model_name)
    model = model.get_model(config)
    return model