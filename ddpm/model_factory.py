from .iddpm import AttnUNet


def create_model(model_name):
    if model_name == 'iddpm':
        return AttnUNet()

