import ast
import inspect
import os
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.ifin import IFINNet
from utils.checkpoint import load_model_state_compat
from utils.operators import gaus_t, generate_roi


@pytest.mark.parametrize("source,options", [
    ("FIXNet.py", {}),
    ("CAW6_buffer.py", dict(seed_blocks="conv", bottleneck=True,
                           residual=True, upsample="bilinear",
                           regularizer_activation="sigmoid")),
])
def test_research_forward_and_gradient_fidelity(source, options):
    root = os.environ.get("IFIN_RESEARCH_MODELS")
    if not root:
        pytest.skip("Set IFIN_RESEARCH_MODELS to compare original research sources")
    tree = ast.parse((Path(root) / source).read_text())
    # Load only definitions; bypass the research module's machine-specific imports.
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    module = types.ModuleType("research_reference")
    module.__dict__.update(torch=torch, nn=nn, F=F, inspect=inspect,
                           gaus_t=gaus_t, generate_roi=generate_roi)
    exec(compile(tree, source, "exec"), module.__dict__)
    torch.manual_seed(31)
    psf = torch.rand(1, 1, 16, 16)
    reference = module.CAWNet(3, 3, psf.clone(), height=32, width=32,
                             dim=4, depth=2, k=4).eval()
    model = IFINNet(3, 3, psf.clone(), height=32, width=32,
                    dim=4, depth=2, k=4, **options).eval()
    load_model_state_compat(model, {"model_state_dict": reference.state_dict()})
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    actual, expected = model(x), reference(y)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    sum(t.square().mean() for t in actual).backward()
    sum(t.square().mean() for t in expected).backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
