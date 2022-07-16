from turtle import forward
from linformer import Linformer
from vit_pytorch.efficient import ViT
from torch import nn
import torch

def load_model(output_size, depth):
    efficient_transformer = Linformer(
        dim=128,
        seq_len=49+1,  # 7x7 patches + 1 cls-token
        depth=depth,
        heads=8,
        k=64
    )

    # =====visual Transformer
    model = ViT(
        dim=128,
        image_size=224,
        patch_size=32,
        num_classes=output_size,
        transformer=efficient_transformer,
        channels=1,
    )
    return model

class SpectralTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(SpectralTransformer, self).__init__()
        self.model=load_model(kwargs['output_size'],kwargs['depth']).to(device=kwargs['device'])
    
    def make_ortho_weights(self,x):
        eps= 1e-7
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_sym = torch.mm(x.t(), x) 
        x_sym += torch.eye(x_sym.shape[1],device=device)*eps
        L = torch.cholesky(x_sym)
        ortho_weights = ((x.shape[0])**(1/2) )* (torch.inverse(L)).t()
        return ortho_weights
    def ortho_update(self,x):
        ortho_weights = self.make_ortho_weights(x)
        self.W_ortho = ortho_weights
    def forward(self, x , ortho_step=False):
        x_net = self.model(x)
        if ortho_step:
            self.ortho_update(x_net)
        y = x_net@self.W_ortho
        return y