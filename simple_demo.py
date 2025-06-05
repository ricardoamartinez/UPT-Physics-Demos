#!/usr/bin/env python3
import sys
import os
sys.path.append('src')

print("=== Universal Physics Transformer (UPT) Simple Demo ===")

# Basic setup
import torch
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Test basic UPT components
try:
    # Initialize static configuration
    from configs.static_config import StaticConfig
    static_config = StaticConfig(uri="static_config.yaml", datasets_were_preloaded=False)
    print("✓ Static configuration initialized successfully")
    
    # Set up dataset configuration provider
    from providers.dataset_config_provider import DatasetConfigProvider
    dataset_config_provider = DatasetConfigProvider(
        global_dataset_paths=static_config.get_global_dataset_paths(),
        local_dataset_path=static_config.get_local_dataset_path(),
        data_source_modes=static_config.get_data_source_modes(),
    )
    print("✓ Dataset configuration provider initialized successfully")
    
    # Import core datasets
    from datasets.lagrangian_dataset import LagrangianDataset
    print("✓ LagrangianDataset imported successfully")
    
    # Try to create a simple dataset instance
    print("\nTesting dataset creation...")
    # This will attempt to download and set up the TGV2D dataset
    dataset = LagrangianDataset(
        name="tgv2d",
        split="train", 
        n_input_timesteps=3,
        n_pushforward_timesteps=9,
        graph_mode="radius_graph_with_supernodes",
        radius_graph_r=0.1,
        radius_graph_max_num_neighbors=4,
        n_supernodes=256,
        num_points_range=[1250, 2500],
        dataset_config_provider=dataset_config_provider,
    )
    
    print(f"✓ Dataset created successfully with {len(dataset)} samples")
    
    # Test loading a sample
    print("\nTesting data loading...")
    # Create a context for the dataset
    ctx = {}
    positions, particle_types = dataset.get_window(0, ctx=ctx)
    print(f"✓ Sample loaded successfully")
    print(f"  - Positions shape: {positions.shape}")
    print(f"  - Particle types shape: {particle_types.shape}")
    print(f"  - Number of timesteps: {positions.shape[0]}")
    print(f"  - Number of particles: {positions.shape[1]}")
    print(f"  - Spatial dimensions: {positions.shape[2]}")
    
    print("\n🎉 UPT setup successful! The Universal Physics Transformer is ready to use.")
    print("\nTo run full training, use:")
    print("python src/main_train.py --devices 0 --hp src/yamls/lagrangian/tgv2d.yaml")
    
except Exception as e:
    print(f"❌ Error during demo: {e}")
    print("\nThis might be due to:")
    print("1. Network issues downloading the dataset")
    print("2. Missing dependencies")
    print("3. Configuration issues")
    
    import traceback
    traceback.print_exc()
