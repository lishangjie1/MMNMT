import torch 
import torch.nn as nn
import torch.nn.functional as F
def freeze_non_moe_parameters(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "moe_block" not in n:
            p.requires_grad = False

        if "wte.weight" in n or "lm_head.weight" in n or "fusion" in n: # or "ln" in n:
            p.requires_grad = True

        # if "fusion" in n:
        #     p.requires_grad = True
        

        
        
