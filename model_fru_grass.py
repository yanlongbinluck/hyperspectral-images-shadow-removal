
from torch.nn import Module, Sequential, Linear, LeakyReLU, Sigmoid

class GeneratorResNet(Module):
    """docstring for Net"""
    def __init__(self):
        super(GeneratorResNet, self).__init__()
        self.main_network = Sequential(
            Linear(256,256),
            LeakyReLU(0.1),
            Linear(256,256),
            LeakyReLU(0.1),
            Linear(256,256),
            LeakyReLU(0.1),
            Linear(256,256),
            Sigmoid()
            )
    def forward(self,x):
        x = self.main_network(x)
        return x
        
class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer = Sequential(
            Linear(256, 256),
            LeakyReLU(0.1),
            Linear(256,256),
            LeakyReLU(0.1),
            Linear(256, 16),
            #LeakyReLU(0.1),
            #Linear(16, 1),
            Sigmoid()
            )

    def forward(self, x):
        output = self.layer(x)
        return output
