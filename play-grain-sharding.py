#!/usr/bin/env python3
"""
Grain Multi-GPU Data Sharding Demo

Demonstrates Google's Grain library for distributed data loading across 
multiple GPUs with JAX using actual Grain IndexSampler and DataLoader.

Usage:
  python grain_sharding_demo.py --num_gpus 4 --batch_size 128 --records 10000

Profiling with NVIDIA Nsight Systems:
  nsys profile python grain_sharding_demo.py --num_gpus 4 --batch_size 128
  nsys profile -o grain_profile python grain_sharding_demo.py --args
  nsys profile --trace=cuda,nvtx python grain_sharding_demo.py --args

NCCL Troubleshooting (if needed):
  export NCCL_P2P_DISABLE=1
  python grain_sharding_demo.py --num_gpus 10 --batch_size 320
"""

import jax
import jax.numpy as jnp
from jax import sharding
from typing import Iterator
import time
import argparse

# Import actual Grain library
import grain
from grain.samplers import IndexSampler
from grain.sharding import ShardOptions


class SimpleNeuralNetwork:
    """Two-layer neural network."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int = 42):
        key = jax.random.PRNGKey(seed)
        key1, key2, key3 = jax.random.split(key, 3)

        self.W1 = jax.random.normal(key1, (input_size, hidden_size)) * 0.1
        self.b1 = jax.random.normal(key2, (hidden_size,)) * 0.01
        self.W2 = jax.random.normal(key3, (hidden_size, output_size)) * 0.1
        self.b2 = jnp.zeros((output_size,))

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        h1 = jnp.dot(x, self.W1) + self.b1
        h1_relu = jnp.maximum(0, h1)
        return jnp.dot(h1_relu, self.W2) + self.b2

    def loss(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        predictions = self.forward(x)
        return jnp.mean((predictions - y) ** 2)


def create_device_mesh(num_devices: int) -> sharding.Mesh:
    devices = jax.devices()[:num_devices]
    return sharding.Mesh(devices, axis_names=('data',))


def shard_batch(batch: jnp.ndarray, device_mesh: sharding.Mesh) -> jnp.ndarray:
    batch_sharding = sharding.NamedSharding(device_mesh, sharding.PartitionSpec('data'))
    return jax.device_put(batch, batch_sharding)


def create_dummy_targets(batch_shape: tuple, output_size: int, device_mesh: sharding.Mesh) -> jnp.ndarray:
    global_batch_size = batch_shape[0]
    targets = jax.random.normal(jax.random.PRNGKey(123), (global_batch_size, output_size))
    target_sharding = sharding.NamedSharding(device_mesh, sharding.PartitionSpec('data'))
    return jax.device_put(targets, target_sharding)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Grain-Style Multi-GPU Data Sharding Demo')
    parser.add_argument('--num_gpus', type=int, required=True,
                        help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Per-device batch size')
    parser.add_argument('--records', type=int, default=10000,
                        help='Total records in dataset')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum number of training steps')
    parser.add_argument('--input_dim', type=int, default=128,
                        help='Input dimension')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden layer size')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='Shuffle data (Grain global transformation)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== Grain-Style Multi-GPU Data Sharding Demo ===")
    print(f"JAX version: {jax.__version__}")

    available_devices = len(jax.devices())
    if args.num_gpus > available_devices:
        print(f"Error: Requested {args.num_gpus} GPUs, only {available_devices} available")
        return

    global_batch_size = args.batch_size * args.num_gpus

    print(f"\nConfiguration:")
    print(f"  Devices: {args.num_gpus}")
    print(f"  Per-device batch size: {args.batch_size}")  
    print(f"  Global batch size: {global_batch_size}")
    print(f"  Total records: {args.records}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Shuffle: {args.shuffle}")
    print(f"  Seed: {args.seed}")

    device_mesh = create_device_mesh(args.num_gpus)

    # Create data using Grain's MapDataset with JAX random
    print(f"\nCreating dataset with {args.records} records...")
    key = jax.random.PRNGKey(42)
    dummy_data = []
    for i in range(args.records):
        key, subkey = jax.random.split(key)
        sample = jax.random.normal(subkey, (args.input_dim,), dtype=jnp.float32)
        dummy_data.append(sample)
    data_source = grain.MapDataset.source(dummy_data)
    print(f"Dataset created with {len(data_source)} records")

    # Create Grain data loaders with proper sharding
    records_per_shard = args.records // args.num_gpus  # With drop_remainder=True
    batches_per_shard = records_per_shard // args.batch_size
    
    print(f"\nSharding details:")
    print(f"  Records per GPU: {records_per_shard}")
    print(f"  Batches per GPU per epoch: {batches_per_shard}")
    
    data_loaders = []
    for device_idx in range(args.num_gpus):
        # Create IndexSampler with ShardOptions for this device
        sampler = IndexSampler(
            num_records=args.records,
            shard_options=ShardOptions(
                shard_index=device_idx,
                shard_count=args.num_gpus,
                drop_remainder=True
            ),
            shuffle=args.shuffle,
            num_epochs=None,  # Infinite epochs for iteration-based training
            seed=args.seed
        )

        # Create Grain DataLoader with batching operation
        data_loader = grain.DataLoader(
            data_source=data_source,
            sampler=sampler,
            operations=[grain.transforms.Batch(batch_size=args.batch_size)],
            worker_count=0,  # Single process for simplicity
        )
        data_loaders.append(data_loader)

        print(f"Shard {device_idx}: {records_per_shard} records, {batches_per_shard} batches/epoch")

    model = SimpleNeuralNetwork(
        input_size=args.input_dim,
        hidden_size=args.hidden_size,
        output_size=10
    )

    @jax.jit
    def compute_loss(x_batch, y_batch):
        return model.loss(x_batch, y_batch)

    print(f"\nStarting iteration-based training...")
    
    # Create device iterators for infinite data stream
    device_iterators = [iter(loader) for loader in data_loaders]
    
    # Training loop with iteration-based progress
    start_time = time.time()
    for step in range(args.max_steps):
        # Get batches from all devices
        device_batches = [next(device_iter) for device_iter in device_iterators]
        global_batch = jnp.concatenate(device_batches, axis=0)
        x_sharded = shard_batch(global_batch, device_mesh)
        y_sharded = create_dummy_targets(global_batch.shape, 10, device_mesh)

        # Compute loss
        loss_value = compute_loss(x_sharded, y_sharded)
        
        # Log progress every 10 steps
        if (step + 1) % 10 == 0 or step == 0:
            # Calculate current epoch (derived when needed)
            samples_processed = (step + 1) * global_batch_size
            current_epoch = samples_processed / args.records
            elapsed_time = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed_time if elapsed_time > 0 else 0
            
            print(f"Step {step + 1}/{args.max_steps} (epoch {current_epoch:.2f}): "
                  f"loss={loss_value:.4f}, {steps_per_sec:.1f} steps/s")

    print(f"\nTraining completed successfully!")


if __name__ == "__main__":
    main()
