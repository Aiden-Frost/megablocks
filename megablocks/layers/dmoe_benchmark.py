import unittest
from functools import partial

from absl.testing import parameterized
from megablocks import grouped_gemm_util as gg
from megablocks.layers.arguments import Arguments
from megablocks.layers import dmoe, testing
from megablocks.layers import moe
import torch

from dmoe_groupedBatched import groupedBatchedDMOE
def test_modules(
        hidden_size,
        ffn_hidden_size,
        moe_num_experts=1,
        moe_capacity_factor=1,
        moe_top_k=1,
        mlp_impl='groupedBatched'):
    init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.1)
    args = Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=moe_num_experts,
        moe_capacity_factor=moe_capacity_factor,
        moe_top_k=moe_top_k,
        init_method=init_method,
        memory_optimized_mlp=False,
        mlp_type='mlp',
        mlp_impl=mlp_impl,
        fp16=True,
        bf16=False)

    if mlp_impl == "groupedBatched":
        dmoe_mlp = groupedBatchedDMOE(args)
    else:
        dmoe_mlp = dmoe.dMoE(args)

    dmoe_mlp.cuda(torch.cuda.current_device()).to(torch.half)
    return args, dmoe_mlp

# min size: (1, 2, 128, 2, 1)
_FORWARD_TESTS_DEFAULT = (
    (16, 1024, 512, 1, 1),
    (16, 1024, 512, 2, 1),
    (16, 1024, 512, 4, 1),
    (16, 1024, 512, 8, 1),
    (8, 2048, 512, 1, 1),
    (8, 2048, 512, 2, 1),
    (8, 2048, 512, 4, 1),
    (16, 1024, 512, 2, 2),
    (16, 1024, 512, 4, 2),
    (16, 1024, 512, 4, 4),
    (16, 1024, 512, 8, 2),
    (16, 1024, 512, 8, 4),
    (16, 1024, 512, 8, 8),
    (32, 1024, 512, 8, 8),
    (64, 1024, 512, 8, 8),
    (64, 1024, 8192, 8, 8),
)

_FORWARD_TESTS = (_FORWARD_TESTS_DEFAULT)

class dMoETest(parameterized.TestCase):

    @staticmethod
    def tearDown():
        moe.clear_load_balancing_loss()

    @parameterized.parameters(*_FORWARD_TESTS)
    def testdMoE_Forward(self, bs, sl, hs, num_experts, top_k,
                         mlp_impl='groupedBatched'):
        x = torch.randn(sl, bs, hs).to(torch.half).cuda()

        _, layer = test_modules(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            moe_num_experts=num_experts,
            moe_top_k=top_k,
            mlp_impl=mlp_impl)

        out, _ = layer(x)
        self.assertSequenceEqual(out.shape, x.shape)

if __name__ == '__main__':
    unittest.main()
