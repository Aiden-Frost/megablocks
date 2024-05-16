from megablocks.layers import common
from megablocks.layers import mpu
from megablocks.layers import router
from megablocks.layers.arguments import Arguments
import megablocks.ops as ops
import numpy as np
import torch
from mlp_groupedBatched import GroupedBatchedMLP
from megablocks.layers import moe


class groupedBatchedExperts(torch.nn.Module):
    def __init__(self, args: Arguments):
        super(groupedBatchedExperts, self).__init__(args)
        self.args = args
        self.hidden_size = args.hidden_size
        self.ffn_hidden_size = mpu.features_per_rank(args)
        self.blocking = 128
        self.mlp = GroupedBatchedMLP(args)
        self.sort_end_bit = max(int(np.ceil(np.log2(args.moe_num_experts))), 1)
        if self.args.bias:
            self.bias = torch.nn.Parameter(torch.empty(
                args.hidden_size,
                device=args.device,
                dtype=common.dtype(args)))
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def indices_and_bins(self, top_expert):
        top_expert = top_expert.int()
        bin_ids, indices = ops.sort(top_expert, self.sort_end_bit)

        tokens_per_expert = ops.histogram(top_expert, self.args.moe_num_experts)

        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.view(1) if not len(bins.size()) else bins
        return indices, bin_ids, bins, tokens_per_expert

    def forward(self, x, scores, expert_weights, top_experts):

        in_shape = x.size()

        expert_weights = expert_weights.flatten()
        top_experts = top_experts.flatten()

        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = (
                self.indices_and_bins(top_experts))

        x = x.view(-1, x.shape[-1])
        x = ops.gather(
            x,
            indices,
            bin_ids,
            bins,
            self.args.moe_top_k)

        # Perform the expert computation.
        x = self.mlp(x, tokens_per_expert)

        # Un-route the data for the MoE output.
        out = ops.scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            self.args.moe_top_k)
        x, tokens_per_expert = out, tokens_per_expert

        if self.training:
            moe.save_load_balancing_loss((tokens_per_expert, scores))
        x = x.view(in_shape)
        if self.bias is not None:
            if self.args.return_bias:
                return x, self.bias
            return x + self.bias
        return x


class groupedBatchedDMOE(torch.nn.Module):

    def __init__(self, args: Arguments):
        super(groupedBatchedDMOE, self).__init__()

        # Token router.
        self.router = router.LearnedRouter(args)

        # Expert computation helper.
        self.experts = groupedBatchedExperts(args)

    def forward(self, x):
        # NOTE: If we're going to cast the activations to lower precision
        # do it before we permute the tokens to save bandwidth.
        x = common.cast_if_autocast_enabled(x)

        # Compute the expert scores and assignments.
        scores, expert_weights, top_experts = self.router(x)

        # Compute the experts.
        return self.experts(x, scores, expert_weights, top_experts)
