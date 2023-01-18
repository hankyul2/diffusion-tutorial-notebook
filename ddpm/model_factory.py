from .iddpm import AttnUNet


def create_model(model_name, dropout=0):
    if model_name == 'iddpm':
        return AttnUNet(drop=dropout)

