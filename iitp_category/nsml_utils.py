import os

import nsml
import torch

from utils import inference


def bind_model(model):
    def load(save_folder, **kwargs):
        filename = os.path.join(save_folder, 'model')
        state = torch.load(filename)
        model.load_state_dict(state, strict=True)

    def save(save_folder, **kwargs):
        filename = os.path.join(save_folder, 'model')
        torch.save(model.state_dict(), filename)
        print('Model saved')

    def infer(data_path, **kwargs):
        return inference(model, data_path)

    nsml.bind(save=save, load=load, infer=infer)
