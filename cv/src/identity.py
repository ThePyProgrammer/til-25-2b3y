import torch

class Identity(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return args
    
    def predict(self, *args, **kwargs):
        return args
