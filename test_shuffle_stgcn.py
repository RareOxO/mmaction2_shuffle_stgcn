"""
Lightweight validation script for shuffle_stgcn.
Tests model forward/backward pass with fake data on CPU.
No large GPU memory usage.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

print("=== Test 1: shuffle_tcn channel shuffle ===")
from mmaction.models.utils.gcn_utils import shuffle_tcn

# Small test: in_channels=64, out_channels=64, groups=2
stcn = shuffle_tcn(64, 64, groups=2)
stcn.eval()
x = torch.randn(2, 64, 20, 17)  # (N, C, T, V)
with torch.no_grad():
    out = stcn(x)
print(f"  Input:  {x.shape}")
print(f"  Output: {out.shape}")
assert out.shape == (2, 64, 20, 17), f"Shape mismatch: {out.shape}"
print("  PASSED")

print()
print("=== Test 2: channel shuffle correctness ===")
# channels should be shuffled across groups
x_test = torch.zeros(1, 4, 1, 1)
x_test[0, 0] = 1.0  # group0, ch0
x_test[0, 1] = 2.0  # group0, ch1
x_test[0, 2] = 3.0  # group1, ch0
x_test[0, 3] = 4.0  # group1, ch1
shuffled = shuffle_tcn._channel_shuffle(x_test, groups=2)
# After shuffle: [ch0_g0, ch0_g1, ch1_g0, ch1_g1] = [1,3,2,4]
expected = torch.tensor([1.0, 3.0, 2.0, 4.0])
assert torch.allclose(shuffled[0, :, 0, 0], expected), \
    f"Shuffle wrong: {shuffled[0,:,0,0]}"
print("  PASSED")

print()
print("=== Test 3: STGCN backbone with shuffle_tcn ===")
from mmaction.models.backbones.stgcn import STGCN

model = STGCN(
    graph_cfg=dict(layout='coco', mode='spatial'),
    gcn_adaptive='init',
    gcn_with_res=True,
    tcn_type='shuffle_tcn',
)
model.eval()

# Small batch: N=2, M=2, T=20, V=17, C=3
x = torch.randn(2, 2, 20, 17, 3)
with torch.no_grad():
    out = model(x)
print(f"  Input:  {x.shape}")
print(f"  Output: {out.shape}")
# Expected: (N, M, 256, T_reduced, V)
assert out.shape[0] == 2 and out.shape[1] == 2 and out.shape[2] == 256
print("  PASSED")

print()
print("=== Test 4: Full RecognizerGCN forward pass ===")
from mmaction.registry import MODELS
from mmengine.registry import DefaultScope
DefaultScope.get_instance('test', scope_name='mmaction')

model_cfg = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        graph_cfg=dict(layout='coco', mode='spatial'),
        tcn_type='shuffle_tcn',
    ),
    cls_head=dict(type='GCNHead', in_channels=256, num_classes=60),
)
model = MODELS.build(model_cfg)
model.eval()

# Input shape: (B, num_clips, M, T, V, C)
# For coco 2D keypoints: C=3 (x, y, score)
from mmaction.structures import ActionDataSample
import torch

# B=2, num_clips=1, M=2, T=100, V=17, C=3
inputs = torch.randn(2, 1, 2, 100, 17, 3)

data_samples = []
for i in range(2):
    ds = ActionDataSample()
    ds.set_gt_label(torch.tensor(i % 60))
    data_samples.append(ds)

with torch.no_grad():
    result = model(inputs, data_samples, mode='predict')
print(f"  Predictions: {len(result)} samples")
print(f"  Score shape: {result[0].pred_score.shape}")
assert result[0].pred_score.shape == (60,)
print("  PASSED")

print()
print("=== Test 5: Backward pass (gradient check) ===")
model.train()
inputs_train = torch.randn(2, 1, 2, 100, 17, 3)
data_samples_train = []
for i in range(2):
    ds = ActionDataSample()
    ds.set_gt_label(torch.tensor(i % 60))
    data_samples_train.append(ds)

loss_dict = model(inputs_train, data_samples_train, mode='loss')
total_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
total_loss.backward()
print(f"  Loss: {total_loss.item():.4f}")
# Check gradients exist
grad_norm = sum(
    p.grad.norm().item()
    for p in model.parameters()
    if p.grad is not None
)
print(f"  Gradient norm: {grad_norm:.4f}")
assert grad_norm > 0, "No gradients!"
print("  PASSED")

print()
print("=== All tests PASSED ===")
