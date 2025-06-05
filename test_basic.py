#!/usr/bin/env python3
import sys
import os
sys.path.append('src')

print("Testing basic imports...")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} imported successfully")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import torch_geometric
    print(f"✓ PyTorch Geometric imported successfully")
except Exception as e:
    print(f"✗ PyTorch Geometric import failed: {e}")

try:
    import torch_scatter
    import torch_cluster
    print(f"✓ torch_scatter and torch_cluster imported successfully")
except Exception as e:
    print(f"✗ torch_scatter/torch_cluster import failed: {e}")

try:
    from datasets.lagrangian_dataset import LagrangianDataset
    print(f"✓ LagrangianDataset imported successfully")
except Exception as e:
    print(f"✗ LagrangianDataset import failed: {e}")

print("Basic import test completed!")
