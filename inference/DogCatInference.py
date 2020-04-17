import torch
from ..base.inference import BaseInference


class CatDogInference(BaseInference):

    def __init__(self, model, device, transform = None):
        super(CatDogInference,self).__init__(model,device)
        self.transform = transform

    def inference(self, input_tensor):
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
        logits = self.model(input_tensor)
        classes = torch.argmax(logits,dim=-1).detach().cpu().numpy().tolist()
        return classes
