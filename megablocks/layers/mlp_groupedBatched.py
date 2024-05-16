from megablocks.layers import common
from megablocks.layers import mpu
from megablocks.layers.arguments import Arguments
from megablocks import grouped_gemm_util as gg
import torch
from packaging import version


class ScaleGradient(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        return grad * ctx.scale, None


scale_gradient = ScaleGradient.apply


def resolve_dtensor(weight):
    if version.parse(torch.__version__) >= version.parse('2.0.0'):
        from torch.distributed._tensor import DTensor
        if isinstance(weight, DTensor):
            return weight.to_local()
    return weight


class GroupedBatchedMLP(torch.nn.Module):
    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args
        self._num_rows_per_rank = (
                (mpu.experts_per_rank(args) * mpu.features_per_rank(args)) //
                mpu.get_weight_parallel_world_size(args)
        )

        self.w1 = torch.nn.Parameter(torch.empty(
            self._num_rows_per_rank,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))
        self.w2 = torch.nn.Parameter(torch.empty(
            self._num_rows_per_rank,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))

    def forward(self, x, tokens_per_expert):
        batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1, w2 = (self.w1, self.w2)

        # Re-shape the weights for the grouped GEMMs.
        ne = mpu.experts_per_rank(self.args)
        w1 = resolve_dtensor(w1).view(ne, -1, self.args.hidden_size)
        w2 = resolve_dtensor(w2).view(ne, -1, self.args.hidden_size)

        # Compute the MLP.
        x = gg.ops.gmm(x, w1, batch_sizes, trans_b=True)
        x = self.args.activation_fn(x)
        return gg.ops.gmm(x, w2, batch_sizes)
