# %%
# Force JAX to use CPU
import os

os.environ["JAX_PLATFORM_NAME"] = "gpu"

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import diffrax
import time
from sde.jax.train import build_data_and_model

# %%
# Configuration settings
context_length = 25  # First N steps as observations
forecast_length = 25  # Number of steps to predict
batch_size = 100  # Batch size for timing measurements

# Master random seed for reproducibility
master_seed = 42

# Model configuration (using one of the models from your list)
model_info = {"name": "classic-sunset-19", "kl_weight": 0.1}

# Define the dataset and model configuration
dataset = "mmnist"
white = True
num_latents = 10
num_contents = 64
num_features = 64
num_k = 5
gamma_max = 20.0
int_sub_steps = 3

# Set the solver
solver = diffrax.StratonovichMilstein()


# %%
def load_model_and_params(model_name):
    """Load model parameters from the specified path"""

    params_path = f"./saved_params/{model_name}/params_latest.p"

    # Load the saved parameters
    with open(params_path, "rb") as f:
        params = pickle.load(f)

    # Rebuild the model with the same configuration
    ts, dt, data_train, data_val, model, _ = build_data_and_model(
        dataset=dataset,
        white=white,
        num_latents=num_latents,
        num_contents=num_contents,
        num_features=num_features,
        num_k=num_k,
        gamma_max=gamma_max,
        int_sub_steps=int_sub_steps,
    )

    return params, model, dt, data_val


# %%
def batch_reconst_forecast(
    model,
    params,
    frames_context_batch,
    ts_context,
    ts_forecast,
    key,
    dt,
    solver,
):
    """
    Batched version of forecast generation using vmap.

    Args:
        model: The VideoSDE model
        params: Trained parameters
        frames_context_batch: Batch of context frames [batch_size, context_length, h, w, c]
        ts_context: Timestamps for context frames
        ts_forecast: Timestamps for forecast frames
        key: JAX random key
        dt: Time step for integration
        solver: Diffrax solver

    Returns:
        Batch of forecast frame sequences
    """

    def single_forecast(frames_context, single_key):
        """Process a single example from the batch"""
        # Encode the context frames
        h = model.encoder(params, frames_context)
        w = model.content(params, h)
        x0_posterior, h = model.infer(params, h)

        # Sample x0 from posterior
        x0 = x0_posterior.sample(seed=single_key)

        # Create context for the SDE
        context = {"ts": ts_context, "hs": h}

        # Store the original control term
        original_u = model._sde._u

        # Define a context-aware control function
        def context_aware_u(params, t, x, y, args):
            return jax.lax.cond(
                t <= ts_context[-1],
                lambda: original_u(params, t, x, y, args),
                lambda: jnp.zeros_like(y[0]),
            )

        # Temporarily replace the control term
        model._sde._u = context_aware_u

        # Generate prediction
        key_sde, _ = jax.random.split(single_key)
        xs, _ = model.sde(
            params,
            key_sde,
            x0,
            jnp.concatenate([ts_context, ts_forecast]),
            dt,
            solver,
            {"context": context},
        )

        # Restore the original control term
        model._sde._u = original_u

        # Decode only the forecast latent states to frames
        forecast_xs = xs[len(ts_context) :]
        forecast_frames = model.decoder(
            params,
            jnp.concatenate(
                [w[None, :].repeat(len(forecast_xs), axis=0), forecast_xs], axis=-1
            ),
        )

        return forecast_frames

    # Split the key for each batch element
    keys = jax.random.split(key, batch_size)

    # Use vmap to process the entire batch
    batch_forecast = jax.vmap(single_forecast)(frames_context_batch, keys)

    return batch_forecast


# %%
# Load model and parameters
print(f"Loading model: {model_info['name']}")
params, model, dt, data_val = load_model_and_params(model_info["name"])

# Set up data loader to get batch size context_length + forecast_length
data_val.seq_len = context_length + forecast_length

# Create timestamps
ts_full = jnp.arange(context_length + forecast_length) * dt * int_sub_steps
ts_context = ts_full[:context_length]
ts_forecast = ts_full[context_length : context_length + forecast_length]

# Generate a batch of test data
print(f"Generating batch of {batch_size} test sequences...")
key = jax.random.PRNGKey(master_seed)

# Get a batch of test sequences
test_batch = []
for i in range(batch_size):
    # Get a test sequence with different indices and noise
    test_sequence = data_val.get_trajectory_with_different_noise(
        i % len(data_val),  # Cycle through available test data
        master_seed + i,
        context_length,
    )
    test_batch.append(test_sequence[:context_length])  # Only take context frames

# Stack into a batch array
test_batch = jnp.stack(test_batch)  # Shape: [batch_size, context_length, h, w, c]

print(f"Test batch shape: {test_batch.shape}")

# %%
# Create the vmap'd and jit'd forecast function
print("Compiling batched forecast function...")

# JIT compile the batched forecast function
batch_forecast_jitted = jax.jit(
    lambda context_batch, key: batch_reconst_forecast(
        model, params, context_batch, ts_context, ts_forecast, key, dt, solver
    )
)

# Warm-up compilation run
print("Performing warm-up compilation run...")
key, subkey = jax.random.split(key)
_ = batch_forecast_jitted(test_batch, subkey)
print("Compilation complete.")

# %%
# Perform timing measurements
print("\nPerforming timing measurements...")
times = []

for i in range(10):
    key, subkey = jax.random.split(key)

    # Time the forecast
    start_time = time.time()
    forecast_batch = batch_forecast_jitted(test_batch, subkey)
    forecast_batch.block_until_ready()  # Ensure computation is complete
    end_time = time.time()

    elapsed_time = end_time - start_time
    times.append(elapsed_time)

    print(f"Run {i + 1}/10: {elapsed_time:.4f} seconds")

# Calculate statistics
times_array = np.array(times)
mean_time = np.mean(times_array)
std_time = np.std(times_array)

print("\nTiming Results:")
print(f"Mean time: {mean_time:.4f} seconds")
print(f"Std dev: {std_time:.4f} seconds")
print(f"Per-sample time: {mean_time / batch_size:.6f} seconds")
print(
    f"FPS (frames per second per sample): {forecast_length / (mean_time / batch_size):.2f}"
)
print(f"Total FPS (all samples): {forecast_length * batch_size / mean_time:.2f}")

# %%
# Verify the output shape
print(f"\nOutput forecast batch shape: {forecast_batch.shape}")
print(f"Expected shape: [{batch_size}, {forecast_length}, height, width, channels]")

# %%
# Additional analysis
min_time = np.min(times_array)
max_time = np.max(times_array)
print("\nDetailed timing statistics:")
print(f"Min time: {min_time:.4f} seconds")
print(f"Max time: {max_time:.4f} seconds")
print(f"Range: {max_time - min_time:.4f} seconds")

# Create a simple bar chart of times
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), times_array)
plt.axhline(y=mean_time, color="r", linestyle="--", label=f"Mean: {mean_time:.4f}s")
plt.axhline(
    y=mean_time + std_time,
    color="g",
    linestyle=":",
    label=f"Mean + Std: {mean_time + std_time:.4f}s",
)
plt.axhline(
    y=mean_time - std_time,
    color="g",
    linestyle=":",
    label=f"Mean - Std: {mean_time - std_time:.4f}s",
)
plt.xlabel("Run Number")
plt.ylabel("Time (seconds)")
plt.title(f"Runtime Measurements for Batch Size {batch_size}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("runtime_measurements.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
print("\nRuntime measurement completed successfully!")
