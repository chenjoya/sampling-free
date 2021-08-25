from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from sampling_free import _C

# TODO: Use JIT to replace CUDA implementation in the future.
class _CELoss(Function):
    @staticmethod
    def forward(ctx, logits, targets):
        ctx.save_for_backward(logits, targets)
        num_classes = logits.shape[1]
        ctx.num_classes = num_classes

        losses = _C.celoss_forward(
            logits, targets, num_classes
        )
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        logits, targets = ctx.saved_tensors
        num_classes = ctx.num_classes
        d_loss = d_loss.contiguous()
        d_logits = _C.celoss_backward(
            logits, targets, d_loss, num_classes
        )
        return d_logits, None, None, None, None

ce_loss_cuda = _CELoss.apply

class CELoss(nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, logits, targets,sum=True):
        loss = ce_loss_cuda(logits, targets)
        return loss.sum() if sum else loss

    def __repr__(self):
        tmpstr = self.__class__.__name__
        return tmpstr