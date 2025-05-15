# %%
# Force JAX to use CPU
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import diffrax
import time
from sde.jax.train import build_data_and_model
import pandas as pd

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
def batch_reconst_forecast_all(
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
    Batched version of forecast generation using vmap - returns all frames.
    """

    def single_forecast(frames_context, single_key):
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
def batch_reconst_forecast_single_timestep(
    model,
    params,
    frames_context_batch,
    ts_context,
    ts_forecast_single,
    key,
    dt,
    solver,
    timestep_idx,
):
    """
    Batched version that forecasts up to a single timestep and decodes only that frame.
    """

    def single_forecast(frames_context, single_key):
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

        # Generate prediction up to the desired timestep only
        key_sde, _ = jax.random.split(single_key)
        xs, _ = model.sde(
            params,
            key_sde,
            x0,
            jnp.concatenate([ts_context, ts_forecast_single]),
            dt,
            solver,
            {"context": context},
        )

        # Restore the original control term
        model._sde._u = original_u

        # Decode only the target timestep frame
        target_x = xs[-1]  # Last timestep is our target
        target_frame = model.decoder(params, jnp.concatenate([w, target_x], axis=-1))

        return target_frame

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
    test_sequence = data_val.get_trajectory_with_different_noise(
        i % len(data_val), master_seed + i, context_length
    )
    test_batch.append(test_sequence[:context_length])

# Stack into a batch array
test_batch = jnp.stack(test_batch)
print(f"Test batch shape: {test_batch.shape}")

# %%
# First, measure the time for predicting all frames at once
print("Compiling and measuring all-frames forecast function...")

batch_forecast_all_jitted = jax.jit(
    lambda context_batch, key: batch_reconst_forecast_all(
        model, params, context_batch, ts_context, ts_forecast, key, dt, solver
    )
)

# Warm-up compilation run for all-frames version
key, subkey = jax.random.split(key)
_ = batch_forecast_all_jitted(test_batch, subkey)
print("All-frames compilation complete.")

# Measure all-frames timing
all_frames_times = []
for i in range(10):
    key, subkey = jax.random.split(key)

    forecast_batch = batch_forecast_all_jitted(test_batch, subkey)

    start_time = time.time()
    forecast_batch = batch_forecast_all_jitted(test_batch, subkey)
    forecast_batch.block_until_ready()
    end_time = time.time()

    elapsed_time = end_time - start_time
    all_frames_times.append(elapsed_time)

all_frames_mean = np.mean(all_frames_times)
all_frames_std = np.std(all_frames_times)

print(f"All-frames forecast: {all_frames_mean:.4f} ± {all_frames_std:.4f} seconds")

# %%
# Now measure individual timestep predictions
print("\nMeasuring individual timestep predictions...")

results_data = []

for t_idx in range(forecast_length):
    print(f"Processing timestep {t_idx + 1}/{forecast_length}...")

    # Get timestamps up to this forecast step
    ts_forecast_single = ts_forecast[: t_idx + 1]

    # Create JIT compiled function for this timestep
    batch_forecast_single_jitted = jax.jit(
        lambda context_batch, key: batch_reconst_forecast_single_timestep(
            model,
            params,
            context_batch,
            ts_context,
            ts_forecast_single,
            key,
            dt,
            solver,
            t_idx,
        )
    )

    # Warm-up compilation run
    key, subkey = jax.random.split(key)
    _ = batch_forecast_single_jitted(test_batch, subkey)

    # Measure timing
    timestep_times = []
    for i in range(10):
        key, subkey = jax.random.split(key)

        start_time = time.time()
        forecast_single = batch_forecast_single_jitted(test_batch, subkey)
        forecast_single.block_until_ready()
        end_time = time.time()

        elapsed_time = end_time - start_time
        timestep_times.append(elapsed_time)

    timestep_mean = np.mean(timestep_times)
    timestep_std = np.std(timestep_times)

    results_data.append(
        {
            "Timestep": t_idx + 1,
            "Mean Time (s)": timestep_mean,
            "Std Dev (s)": timestep_std,
            "Per Sample (ms)": timestep_mean * 1000 / batch_size,
            "Speedup vs All": all_frames_mean / timestep_mean,
        }
    )

# %%
# Create and display results table
results_df = pd.DataFrame(results_data)

# Add the all-frames result as a comparison row
all_frames_row = pd.DataFrame(
    [
        {
            "Timestep": "All (1-25)",
            "Mean Time (s)": all_frames_mean,
            "Std Dev (s)": all_frames_std,
            "Per Sample (ms)": all_frames_mean * 1000 / batch_size,
            "Speedup vs All": 1.0,
        }
    ]
)

# Combine the dataframes
full_results_df = pd.concat([results_df, all_frames_row], ignore_index=True)

# Display the table
print("\n" + "=" * 80)
print("RUNTIME MEASUREMENT RESULTS")
print("=" * 80)
print(
    f"Batch size: {batch_size} | Context length: {context_length} | Forecast length: {forecast_length}"
)
print("-" * 80)
print(full_results_df.to_string(index=False, float_format="%.4f"))
print("=" * 80)

# %%
# Create visualization
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Runtime vs Timestep
timesteps = results_df["Timestep"].values
mean_times = results_df["Mean Time (s)"].values
std_times = results_df["Std Dev (s)"].values

ax1.errorbar(timesteps, mean_times, yerr=std_times, fmt="o-", label="Single timestep")
ax1.axhline(
    y=all_frames_mean,
    color="r",
    linestyle="--",
    label=f"All frames: {all_frames_mean:.4f}s",
)
ax1.fill_between(
    [0, 26],
    [all_frames_mean - all_frames_std] * 2,
    [all_frames_mean + all_frames_std] * 2,
    color="red",
    alpha=0.2,
)
ax1.set_xlabel("Forecast Timestep")
ax1.set_ylabel("Time (seconds)")
ax1.set_title("Runtime for Individual Timestep vs All Frames")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 26)

# Plot 2: Speedup factor
speedups = results_df["Speedup vs All"].values
ax2.plot(timesteps, speedups, "g^-", linewidth=2, markersize=8)
ax2.axhline(y=1.0, color="k", linestyle=":", label="No speedup")
ax2.set_xlabel("Forecast Timestep")
ax2.set_ylabel("Speedup Factor")
ax2.set_title("Speedup: All Frames / Single Timestep")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 26)

plt.tight_layout()
plt.savefig("timestep_runtime_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Summary statistics
print("\nSUMMARY STATISTICS:")
print(f"All-frames prediction: {all_frames_mean:.4f} ± {all_frames_std:.4f} seconds")
print(f"Average single-timestep: {results_df['Mean Time (s)'].mean():.4f} seconds")
print(
    f"Fastest single-timestep: {results_df['Mean Time (s)'].min():.4f} seconds (timestep {results_df.loc[results_df['Mean Time (s)'].idxmin(), 'Timestep']})"
)
print(
    f"Slowest single-timestep: {results_df['Mean Time (s)'].max():.4f} seconds (timestep {results_df.loc[results_df['Mean Time (s)'].idxmax(), 'Timestep']})"
)
print(f"Average speedup: {results_df['Speedup vs All'].mean():.2f}x")
print(
    f"Max speedup: {results_df['Speedup vs All'].max():.2f}x (timestep {results_df.loc[results_df['Speedup vs All'].idxmax(), 'Timestep']})"
)
