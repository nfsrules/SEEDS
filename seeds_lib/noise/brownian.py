import torch


class Standard(object):
    def __call__(self, x):
        return torch.randn_like(x)

    
class Discrete(object):
    def __call__(self ,x):
        return (torch.randn_like(x)>0.5)*2.-1.

