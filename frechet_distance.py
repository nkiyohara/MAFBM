# %%
# Force JAX to use CPU
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import diffrax
from tqdm import tqdm
from sde.jax.train import build_data_and_model
import torch
from srvp_fd import FrechetDistanceCalculator
import pandas as pd
from io import BytesIO
from PIL import Image
from moviepy.editor import ImageSequenceClip

# %%
# Configuration settings
context_length = 25  # First N steps as observations
forecast_length = 25  # Number of steps to predict
output_dir = "sde_prediction_results_multi_traj"

# Number of different input trajectories to evaluate
num_trajectories = 5
# Number of samples per trajectory for Monte Carlo estimation
num_samples_per_trajectory = 512

# Visualization trajectory index (dataset index to use for visualization)
viz_trajectory_idx = 10

# Master random seed for reproducibility
master_seed = 42


# Create a single instance to be used across all models
fd_calculator = FrechetDistanceCalculator(
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Define the models to analyze (with different KL weights)
models = [
    # {"name": "earnest-paper-16", "kl_weight": 2.0},    # Still running - commented out
    # {"name": "pious-morning-15", "kl_weight": 1.0},    # Still running - commented out
    # {"name": "effortless-meadow-14", "kl_weight": 0.125},
    # {"name": "graceful-sky-13", "kl_weight": 0.25},
    # {"name": "treasured-river-12", "kl_weight": 0.5}
    {"name": "classic-sunset-19", "kl_weight": 0.1},
    # {"name": "stoic-mountain-18", "kl_weight": 1.0},
]

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the dataset and model configuration
# These should match the configuration used during training
dataset = "mmnist"  # or "ball", "bar", "bounce" depending on what was used
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
def reconst_forecast(
    model,
    params,
    frames_context,
    ts_context,
    ts_forecast,
    key,
    dt,
    solver,
    num_samples=1,
):
    """
    Generate forecasts based on context frames.

    Args:
        model: The VideoSDE model
        params: Trained parameters
        frames_context: Context frames to condition on
        ts_context: Timestamps for context frames
        ts_forecast: Timestamps for forecast frames
        key: JAX random key
        dt: Time step for integration
        solver: Diffrax solver
        num_samples: Number of forecast samples to generate

    Returns:
        List of forecast frame sequences
    """
    # Encode the context frames
    h = model.encoder(params, frames_context)
    w = model.content(params, h)
    x0_posterior, h = model.infer(params, h)

    # Generate samples
    reconsts = []
    forecasts = []
    for i in range(num_samples):
        key, sample_key = jax.random.split(key)

        # Sample x0 from posterior
        x0 = x0_posterior.sample(seed=sample_key)

        # Create context for the SDE
        context = {"ts": ts_context, "hs": h}

        # Store the original control term
        original_u = model._sde._u

        # Define a context-aware control function
        def context_aware_u(params, t, x, y, args):
            return jax.lax.cond(
                t <= ts_context[-1],
                # True branch: use original control for context period
                lambda: original_u(params, t, x, y, args),
                # False branch: use zero control for forecast period
                lambda: jnp.zeros_like(y[0]),
            )

        # Temporarily replace the control term
        model._sde._u = context_aware_u

        # Generate prediction
        key, forecast_key = jax.random.split(key)
        xs, _ = model.sde(
            params,
            forecast_key,
            x0,
            jnp.concatenate([ts_context, ts_forecast]),
            dt,
            solver,
            {"context": context},
        )

        # Restore the original control term
        model._sde._u = original_u

        # Decode the latent states to frames
        frames = model.decoder(
            params, jnp.concatenate([w[None, :].repeat(len(xs), axis=0), xs], axis=-1)
        )
        reconsts.append(frames[: len(ts_context)])
        forecasts.append(frames[len(ts_context) :])

    return reconsts, forecasts


# %%
def jax_to_torch(jax_array):
    """Convert JAX array to PyTorch tensor for Fréchet distance calculation"""
    # Convert from JAX to numpy first
    x_np = np.array(jax_array)
    # Move channels to second dimension (PyTorch format)
    x_np = np.transpose(x_np, (0, 4, 1, 2, 3))  # [batch, channels, time, height, width]
    # Reshape to [batch*time, channels, height, width]
    batch, channels, time, height, width = x_np.shape
    x_reshaped = x_np.reshape(batch * time, channels, height, width)
    # Convert to PyTorch tensor
    return torch.from_numpy(x_reshaped)


# %%
def analyze_model(model_info, model_idx):
    """Analyze a single model with multiple trajectories"""

    model_name = model_info["name"]
    kl_weight = model_info["kl_weight"]

    print(f"Analyzing model: {model_name} with KL weight = {kl_weight}")

    # Load model and parameters
    params, model, dt, data_val = load_model_and_params(model_name)

    # Set deterministic key based on master seed and model index
    key = jax.random.PRNGKey(master_seed + model_idx * 100)

    # For visualization only - use the specified visualization trajectory
    data_val.seq_len = context_length + forecast_length

    # Get visualization trajectory
    test_sequence = data_val.get_trajectory_with_different_noise(
        viz_trajectory_idx, master_seed, context_length
    )

    frames_full = test_sequence
    frames_context = frames_full[:context_length]
    frames_forecast_gt = frames_full[context_length : context_length + forecast_length]

    # Create timestamps
    ts_full = jnp.arange(len(frames_full)) * dt * int_sub_steps
    ts_context = ts_full[:context_length]
    ts_forecast = ts_full[context_length : context_length + forecast_length]

    # Generate forecasts for visualization
    viz_key, key = jax.random.split(key)
    num_viz_samples = 4
    viz_reconsts, viz_forecasts = reconst_forecast(
        model=model,
        params=params,
        frames_context=frames_context,
        ts_context=ts_context,
        ts_forecast=ts_forecast,
        key=viz_key,
        dt=dt,
        solver=solver,
        num_samples=num_viz_samples,
    )

    # JIT-compiled forecast function for faster sampling
    @jax.jit
    def jitted_forecast(key, context_frames):
        reconsts, forecasts = reconst_forecast(
            model=model,
            params=params,
            frames_context=context_frames,
            ts_context=ts_context,
            ts_forecast=ts_forecast,
            key=key,
            dt=dt,
            solver=solver,
            num_samples=1,
        )
        return forecasts[0]  # Return the single forecast

    # Initialize arrays to store Fréchet distances across all trajectories
    all_traj_fd = []

    # Loop through multiple trajectories for evaluation
    print(f"Evaluating across {num_trajectories} different trajectories...")
    for traj_idx in range(num_trajectories):
        print(f"Processing trajectory {traj_idx + 1}/{num_trajectories}")

        # Deterministic seed for each trajectory
        traj_seed = master_seed + traj_idx * 1000 + model_idx * 100
        traj_key = jax.random.PRNGKey(traj_seed)

        # Get the trajectory
        test_sequence = data_val.get_trajectory_with_different_noise(
            traj_idx,  # Use different trajectories
            traj_seed,
            context_length,
        )

        frames_full = test_sequence
        frames_context = frames_full[:context_length]

        # Generate model samples for this trajectory
        model_samples = []

        # Generate keys in advance for deterministic results
        sample_keys = jax.random.split(traj_key, num_samples_per_trajectory)

        # Use tqdm to show progress
        for i in tqdm(range(num_samples_per_trajectory)):
            forecast_sample = jitted_forecast(sample_keys[i], frames_context)
            model_samples.append(forecast_sample)

        # Stack all forecasts for easier computation
        model_samples = jnp.stack(model_samples)

        # Generate ground truth samples
        gt_samples = []
        gt_keys = jax.random.split(
            jax.random.PRNGKey(traj_seed + 500), num_samples_per_trajectory
        )

        for i in tqdm(range(num_samples_per_trajectory)):
            # Get a trajectory with different noise
            traj = data_val.get_trajectory_with_different_noise(
                index=traj_idx,  # Use the same trajectory index
                noise_seed=int(gt_keys[i][0]),  # Use different random seeds
                divergence_step=context_length,  # Start divergence after context frames
            )
            # Extract the forecast part
            forecast_frames = traj[context_length : context_length + forecast_length]
            gt_samples.append(forecast_frames)

        gt_samples = jnp.stack(gt_samples)

        # Calculate per-frame Fréchet distances for this trajectory
        frame_fds = []

        for t in range(forecast_length):
            # Extract samples for frame t
            gt_frame_samples = gt_samples[:, t]  # [n_samples, height, width, channels]
            model_frame_samples = model_samples[
                :, t
            ]  # [n_samples, height, width, channels]

            # Convert to PyTorch tensors
            gt_frame_torch = torch.from_numpy(
                np.transpose(np.array(gt_frame_samples), (0, 3, 1, 2))
            )
            model_frame_torch = torch.from_numpy(
                np.transpose(np.array(model_frame_samples), (0, 3, 1, 2))
            )

            # Calculate Fréchet distance for this frame
            frame_fd = fd_calculator(gt_frame_torch, model_frame_torch)
            frame_fds.append(frame_fd)

        all_traj_fd.append(frame_fds)

    # Calculate average Fréchet distances across all trajectories
    avg_fd = np.mean(np.array(all_traj_fd), axis=0)
    std_fd = np.std(np.array(all_traj_fd), axis=0)

    # For visualization - generate statistics on the visualization trajectory
    # Generate with a fixed key for reproducibility
    viz_stats_key = jax.random.PRNGKey(master_seed + 9999 + model_idx * 100)

    # Generate many samples for the visualization trajectory
    viz_forecast_samples = []
    viz_sample_keys = jax.random.split(viz_stats_key, num_samples_per_trajectory)

    for i in tqdm(range(num_samples_per_trajectory)):
        forecast_sample = jitted_forecast(viz_sample_keys[i], frames_context)
        viz_forecast_samples.append(forecast_sample)

    # Stack all forecasts for easier computation
    viz_all_forecasts = jnp.stack(viz_forecast_samples)

    # Compute mean and standard deviation across samples
    viz_mean_forecast = jnp.mean(viz_all_forecasts, axis=0)
    viz_std_forecast = jnp.std(viz_all_forecasts, axis=0)

    # Generate GT statistics for visualization
    viz_gt_samples = []
    viz_gt_keys = jax.random.split(
        jax.random.PRNGKey(master_seed + 10000 + model_idx * 100),
        num_samples_per_trajectory,
    )

    for i in tqdm(range(num_samples_per_trajectory)):
        # Get a trajectory with different noise
        viz_gt_traj = data_val.get_trajectory_with_different_noise(
            index=viz_trajectory_idx,  # Use visualization trajectory index
            noise_seed=int(viz_gt_keys[i][0]),  # Use different random seeds
            divergence_step=context_length,  # Start divergence after context frames
        )
        # Extract the forecast part
        viz_gt_forecast_frames = viz_gt_traj[
            context_length : context_length + forecast_length
        ]
        viz_gt_samples.append(viz_gt_forecast_frames)

    viz_gt_samples = jnp.stack(viz_gt_samples)
    viz_mean_gt = jnp.mean(viz_gt_samples, axis=0)
    viz_std_gt = jnp.std(viz_gt_samples, axis=0)

    results = {
        "model_name": model_name,
        "kl_weight": kl_weight,
        "ts_context": ts_context,
        "ts_forecast": ts_forecast,
        # Average Fréchet distances across all trajectories
        "avg_fd": avg_fd,
        "std_fd": std_fd,
        # All individual trajectory Fréchet distances
        "all_traj_fd": all_traj_fd,
        # Visualization data
        "viz_frames_context": frames_context,
        "viz_reconsts": viz_reconsts,
        "viz_forecasts": viz_forecasts,
        "viz_mean_forecast": viz_mean_forecast,
        "viz_std_forecast": viz_std_forecast,
        "viz_mean_gt": viz_mean_gt,
        "viz_std_gt": viz_std_gt,
    }

    return results


# %%
# Analyze all models
all_results = []
for i, model_info in enumerate(models):
    results = analyze_model(model_info, i)
    all_results.append(results)


# %%
# Visualization functions
def visualize_reconstruction(results, output_dir=output_dir):
    """Visualize reconstruction of context frames"""

    for result in results:
        model_name = result["model_name"]
        kl_weight = result["kl_weight"]
        frames_context = result["viz_frames_context"]
        reconsts = result["viz_reconsts"]
        ts_context = result["ts_context"]

        fig, axes = plt.subplots(2, context_length, figsize=(2.5 * context_length, 5))

        # Add titles on the left
        axes[0, 0].set_ylabel("Input", size=12)
        axes[1, 0].set_ylabel("Reconstruction", size=12)

        # Add title on top
        fig.suptitle(
            f"Context Frames - {model_name} (KL weight = {kl_weight})", size=14, y=1.05
        )

        for t in range(context_length):
            # Input sequence
            axes[0, t].imshow(frames_context[t], cmap="gray", vmin=0, vmax=1)
            axes[0, t].set_xticks([])
            axes[0, t].set_yticks([])
            axes[0, t].set_title(f"t={ts_context[t]:.2f}")

            # Reconstruction sequence
            axes[1, t].imshow(reconsts[0][t], cmap="gray", vmin=0, vmax=1)
            axes[1, t].set_xticks([])
            axes[1, t].set_yticks([])

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"reconstruction_{model_name}.png"), dpi=150
        )
        plt.close()


def visualize_forecasts(results, output_dir=output_dir):
    """Visualize forecast samples"""

    for result in results:
        model_name = result["model_name"]
        kl_weight = result["kl_weight"]
        forecasts = result["viz_forecasts"]
        ts_forecast = result["ts_forecast"]

        num_forecast_samples = len(forecasts)

        fig, axes = plt.subplots(
            num_forecast_samples,
            forecast_length,
            figsize=(2.5 * forecast_length, 2.5 * num_forecast_samples),
        )

        # Add titles
        for i in range(num_forecast_samples):
            axes[i, 0].set_ylabel(f"Sample {i + 1}", size=12)

        fig.suptitle(
            f"Forecast Frames - {model_name} (KL weight = {kl_weight})", size=14, y=1.05
        )

        for i in range(num_forecast_samples):
            for t in range(forecast_length):
                # Predicted sequence
                axes[i, t].imshow(forecasts[i][t, ..., 0], cmap="gray", vmin=0, vmax=1)
                axes[i, t].set_xticks([])
                axes[i, t].set_yticks([])
                if i == 0:
                    axes[i, t].set_title(f"t={ts_forecast[t]:.2f}")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"forecasts_{model_name}.png"), dpi=150)
        plt.close()


def visualize_forecast_statistics(results, output_dir=output_dir):
    """Visualize mean and std dev of forecasts"""

    for result in results:
        model_name = result["model_name"]
        kl_weight = result["kl_weight"]
        mean_forecast = result["viz_mean_forecast"]
        std_forecast = result["viz_std_forecast"]
        ts_forecast = result["ts_forecast"]

        fig, axes = plt.subplots(2, forecast_length, figsize=(2.5 * forecast_length, 5))

        # Add titles on the left
        axes[0, 0].set_ylabel("Mean", size=12)
        axes[1, 0].set_ylabel("Std Dev", size=12)

        # Add title on top
        fig.suptitle(
            f"Forecast Statistics - {model_name} (KL weight = {kl_weight})",
            size=14,
            y=1.05,
        )

        for t in range(forecast_length):
            # Mean visualization
            im_mean = axes[0, t].imshow(
                mean_forecast[t, ..., 0], cmap="gray", vmin=0, vmax=1
            )
            axes[0, t].set_xticks([])
            axes[0, t].set_yticks([])
            axes[0, t].set_title(f"t={ts_forecast[t]:.2f}")

            # Standard deviation visualization
            im_std = axes[1, t].imshow(
                std_forecast[t, ..., 0], cmap="plasma", vmin=0, vmax=std_forecast.max()
            )
            axes[1, t].set_xticks([])
            axes[1, t].set_yticks([])

        # Add colorbars
        cbar_ax_mean = fig.add_axes([0.92, 0.55, 0.01, 0.3])
        cbar_ax_std = fig.add_axes([0.92, 0.15, 0.01, 0.3])
        fig.colorbar(im_mean, cax=cbar_ax_mean, label="Pixel Value")
        fig.colorbar(im_std, cax=cbar_ax_std, label="Standard Deviation")

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(
            os.path.join(output_dir, f"forecast_stats_{model_name}.png"), dpi=150
        )
        plt.close()


def visualize_gt_statistics(results, output_dir=output_dir):
    """Visualize ground truth mean and std dev"""

    result = results[0]  # Ground truth visualization is the same for all models
    mean_gt = result["viz_mean_gt"]
    std_gt = result["viz_std_gt"]
    ts_forecast = result["ts_forecast"]

    # Visualize ground truth statistics
    fig, axes = plt.subplots(2, forecast_length, figsize=(2.5 * forecast_length, 5))

    fig.suptitle("Ground Truth Statistics", size=14, y=1.05)

    axes[0, 0].set_ylabel("Mean", size=12)
    axes[1, 0].set_ylabel("Std Dev", size=12)

    for t in range(forecast_length):
        im_mean = axes[0, t].imshow(mean_gt[t, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_xticks([])
        axes[0, t].set_yticks([])
        axes[0, t].set_title(f"t={ts_forecast[t]:.2f}")

        im_std = axes[1, t].imshow(
            std_gt[t, ..., 0], cmap="plasma", vmin=0, vmax=std_gt.max()
        )
        axes[1, t].set_xticks([])
        axes[1, t].set_yticks([])

    cbar_ax_mean = fig.add_axes([0.92, 0.55, 0.01, 0.3])
    cbar_ax_std = fig.add_axes([0.92, 0.15, 0.01, 0.3])
    fig.colorbar(im_mean, cax=cbar_ax_mean, label="Pixel Value")
    fig.colorbar(im_std, cax=cbar_ax_std, label="Standard Deviation")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(output_dir, "stats_ground_truth.png"), dpi=150)
    plt.close()


def plot_frechet_distances(results, output_dir=output_dir):
    """Plot average Fréchet distances for all models"""

    plt.figure(figsize=(12, 8))

    # Plot Fréchet distances for each model
    for result in results:
        model_name = result["model_name"]
        kl_weight = result["kl_weight"]
        ts_forecast = result["ts_forecast"]
        avg_fd = result["avg_fd"]

        plt.plot(
            ts_forecast, avg_fd, "-o", label=f"{model_name} (KL weight = {kl_weight})"
        )

    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Fréchet Distance", fontsize=14)
    plt.title(
        f"Average Fréchet Distance across {num_trajectories} Trajectories", fontsize=16
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_frechet_distances.png"), dpi=150)
    plt.close()

    # Plot trajectory-wise comparisons for the first model
    result = results[0]
    model_name = result["model_name"]
    kl_weight = result["kl_weight"]
    ts_forecast = result["ts_forecast"]
    all_traj_fd = result["all_traj_fd"]

    plt.figure(figsize=(10, 6))
    for i, traj_fd in enumerate(all_traj_fd):
        plt.plot(ts_forecast, traj_fd, "-", alpha=0.5, label=f"Trajectory {i + 1}")

    avg_fd = result["avg_fd"]
    plt.plot(ts_forecast, avg_fd, "k-", linewidth=2, label="Average")

    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Fréchet Distance", fontsize=14)
    plt.title(
        f"Trajectory-wise Fréchet Distance ({model_name}, KL weight={kl_weight})",
        fontsize=16,
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"trajectory_comparison_{model_name}.png"), dpi=150
    )
    plt.close()


# %%
# Generate all visualizations
print("Generating visualizations...")

# Visualize reconstructions
visualize_reconstruction(all_results)

# Visualize forecasts
visualize_forecasts(all_results)

# Visualize statistics
visualize_forecast_statistics(all_results)
visualize_gt_statistics(all_results)

# Plot Fréchet distances
plot_frechet_distances(all_results)

print(f"All visualizations saved to {output_dir}")

# %%
# Display a summary of Fréchet distance results
print("Summary of average Fréchet distances across all trajectories:")

for result in all_results:
    model_name = result["model_name"]
    kl_weight = result["kl_weight"]
    fd_mean = np.mean(result["avg_fd"])

    print(f"{model_name} (KL weight = {kl_weight}):")
    print(f"  - Mean Fréchet Distance: {fd_mean:.4f}")

    # Also report early vs late prediction performance
    early_steps = forecast_length // 3  # First third of prediction steps
    late_steps = forecast_length // 3  # Last third of prediction steps

    fd_early = np.mean(result["avg_fd"][:early_steps])
    fd_late = np.mean(result["avg_fd"][-late_steps:])

    print(f"  - Early Fréchet Distance: {fd_early:.4f}")
    print(f"  - Late Fréchet Distance: {fd_late:.4f}")


# %%
# Export numerical results to CSV for further analysis
def export_results_to_csv(results, output_dir=output_dir):
    """Export results to CSV files for further analysis"""

    # Create a DataFrame for the average Fréchet distances
    fd_data = []
    for result in results:
        kl_weight = result["kl_weight"]
        avg_fd = result["avg_fd"]

        for t, fd in enumerate(avg_fd):
            fd_data.append(
                {"KL Weight": kl_weight, "Timestep": t, "Fréchet Distance": fd}
            )

    fd_df = pd.DataFrame(fd_data)
    fd_df.to_csv(os.path.join(output_dir, "avg_frechet_distances.csv"), index=False)

    # Create a summary DataFrame with average FD across all timesteps
    summary_data = []
    for result in results:
        kl_weight = result["kl_weight"]
        avg_fd = result["avg_fd"]

        summary_data.append(
            {
                "KL Weight": kl_weight,
                "Mean FD": np.mean(avg_fd),
                "Min FD": np.min(avg_fd),
                "Max FD": np.max(avg_fd),
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(
        os.path.join(output_dir, "frechet_distance_summary.csv"), index=False
    )

    print(f"Results exported to CSV files in {output_dir}")


def create_prediction_video(results, method="recursive", output_dir=output_dir, fps=5):
    """Create side-by-side video of prediction samples for each model"""

    for i, result in enumerate(results):
        kl_weight = result["kl_weight"]

        # Get prediction samples and times
        pred_samples = result["viz_forecasts"]
        times_pred = result["ts_forecast"]

        # Convert JAX arrays to NumPy arrays if needed
        if hasattr(pred_samples, "device_buffer"):
            pred_samples = np.array(pred_samples)

        # Handle list type prediction samples
        if isinstance(pred_samples, list):
            try:
                pred_samples = np.array(pred_samples)
                print(
                    f"Successfully converted prediction samples to array with shape {pred_samples.shape}"
                )
            except Exception as e:
                print(f"Could not convert prediction samples list to array: {e}")
                continue

        # Create a list to store frames
        frames = []

        # For each timestep, create a frame with 4 samples side by side
        for t in range(forecast_length):
            # Create a figure with 4 samples side by side
            fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))

            # Use actual time value instead of just the index
            actual_time = (
                times_pred[t].item()
                if hasattr(times_pred[t], "item")
                else times_pred[t]
            )
            fig.suptitle(
                f"KL Weight = {kl_weight}, Predictions, Time = {actual_time:.2f}",
                size=14,
            )

            for j in range(min(4, len(pred_samples))):
                try:
                    # Get the prediction frame
                    current_frame = np.array(pred_samples[j][t])

                    # Handle different channel formats
                    if current_frame.ndim == 3 and current_frame.shape[2] == 1:
                        current_frame = current_frame[:, :, 0]

                    # Display the image
                    axes[j].imshow(current_frame, cmap="gray", vmin=0, vmax=1)
                    axes[j].set_xticks([])
                    axes[j].set_yticks([])
                except Exception as e:
                    print(f"Error processing frame {t}, sample {j}: {e}")
                    axes[j].text(
                        0.5,
                        0.5,
                        "Error",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=axes[j].transAxes,
                        color="red",
                    )
                    axes[j].set_xticks([])
                    axes[j].set_yticks([])

            plt.tight_layout()

            # Convert the figure to an image array using PIL
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img = np.array(Image.open(buf))
            frames.append(img)

            plt.close()
            buf.close()

        # Create and save the video
        clip = ImageSequenceClip(frames, fps=fps)
        video_path = os.path.join(output_dir, f"predictions_kl{kl_weight}_video.mp4")
        clip.write_videofile(video_path, codec="libx264", audio=False)

        print(f"Video created for model with KL weight {kl_weight} at {video_path}")


def create_statistics_video(results, output_dir=output_dir, fps=5):
    """Create side-by-side video of mean and variance for each model"""

    for i, result in enumerate(results):
        kl_weight = result["kl_weight"]
        times_pred = result["ts_forecast"]

        # Get the statistics data
        mean_data = result["viz_mean_forecast"]
        std_data = result["viz_std_forecast"]

        # Convert JAX arrays to NumPy arrays if needed
        if hasattr(mean_data, "device_buffer"):
            mean_data = np.array(mean_data)
        if hasattr(std_data, "device_buffer"):
            std_data = np.array(std_data)

        # Create a list to store frames
        frames = []

        # For each timestep, create a frame with mean and std side by side
        for t in range(forecast_length):
            # Create a figure with mean and std side by side
            fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))

            # Use actual time value instead of just the index
            actual_time = (
                times_pred[t].item()
                if hasattr(times_pred[t], "item")
                else times_pred[t]
            )
            fig.suptitle(
                f"KL Weight = {kl_weight}, Statistics, Time = {actual_time:.2f}",
                size=14,
            )

            try:
                # Get the mean frame
                mean_frame = np.array(mean_data[t])

                # Handle different channel formats
                if mean_frame.ndim == 3 and mean_frame.shape[2] == 1:
                    mean_frame = mean_frame[:, :, 0]

                # Display the mean image
                axes[0].imshow(mean_frame, cmap="gray", vmin=0, vmax=1)
                axes[0].set_title("Mean")
                axes[0].set_xticks([])
                axes[0].set_yticks([])

                # Get the std frame
                std_frame = np.array(std_data[t])

                # Handle different channel formats
                if std_frame.ndim == 3 and std_frame.shape[2] == 1:
                    std_frame = std_frame[:, :, 0]

                # Display the std image
                axes[1].imshow(std_frame, cmap="viridis")
                axes[1].set_title("Std Dev")
                axes[1].set_xticks([])
                axes[1].set_yticks([])
            except Exception as e:
                print(f"Error processing statistics frame {t}: {e}")
                for ax in axes:
                    ax.text(
                        0.5,
                        0.5,
                        "Error",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                        color="red",
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

            plt.tight_layout()

            # Convert the figure to an image array using PIL
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img = np.array(Image.open(buf))
            frames.append(img)

            plt.close()
            buf.close()

        # Create and save the video
        clip = ImageSequenceClip(frames, fps=fps)
        video_path = os.path.join(output_dir, f"statistics_kl{kl_weight}_video.mp4")
        clip.write_videofile(video_path, codec="libx264", audio=False)

        print(
            f"Statistics video created for model with KL weight {kl_weight} at {video_path}"
        )


def debug_prediction_samples(results):
    """Debug function to inspect prediction samples before creating videos"""
    print("\n=== Debugging Prediction Samples ===")

    for i, result in enumerate(results):
        kl_weight = result["kl_weight"]
        context_frames = result["viz_frames_context"]
        pred_samples = result["viz_forecasts"]

        # Convert JAX arrays to NumPy arrays if needed
        if hasattr(context_frames, "device_buffer"):
            context_frames = np.array(context_frames)

        print(f"\nModel with KL weight {kl_weight}:")

        # Debug context frames
        print("  Context frames:")
        print(f"  - Type: {type(context_frames)}")
        print(f"  - Shape: {context_frames.shape}")
        print(f"  - Data type: {context_frames.dtype}")
        print(f"  - Min value: {np.min(context_frames)}")
        print(f"  - Max value: {np.max(context_frames)}")
        print(f"  - Mean value: {np.mean(context_frames)}")

        # Check if context frames are mostly zeros or ones
        context_zeros = np.sum(context_frames < 0.01) / context_frames.size * 100
        context_ones = np.sum(context_frames > 0.99) / context_frames.size * 100
        print(f"  - Percentage of near-zero values: {context_zeros:.2f}%")
        print(f"  - Percentage of near-one values: {context_ones:.2f}%")

        # Debug prediction samples
        print("\n  Prediction samples:")
        print(f"  - Type: {type(pred_samples)}")

        # Handle different types of prediction samples
        if isinstance(pred_samples, list):
            print(f"  - List length: {len(pred_samples)}")
            if len(pred_samples) > 0:
                first_sample = pred_samples[0]
                print(f"  - First sample type: {type(first_sample)}")
                if hasattr(first_sample, "shape"):
                    print(f"  - First sample shape: {first_sample.shape}")

                # Convert list to numpy array if possible
                try:
                    pred_samples_array = np.array(pred_samples)
                    print(
                        f"  - Converted to array with shape: {pred_samples_array.shape}"
                    )
                except Exception as e:
                    print(f"  - Could not convert list to numpy array: {e}")
        else:
            # Already an array-like object
            print(f"  - Shape: {pred_samples.shape}")
            print(f"  - Data type: {pred_samples.dtype}")
            print(f"  - Min value: {np.min(pred_samples)}")
            print(f"  - Max value: {np.max(pred_samples)}")
            print(f"  - Mean value: {np.mean(pred_samples)}")

            # Check if prediction samples are mostly zeros or ones
            zeros_percentage = np.sum(pred_samples < 0.01) / pred_samples.size * 100
            ones_percentage = np.sum(pred_samples > 0.99) / pred_samples.size * 100
            print(f"  - Percentage of near-zero values: {zeros_percentage:.2f}%")
            print(f"  - Percentage of near-one values: {ones_percentage:.2f}%")

        # Check a sample frame from context
        if context_frames.shape[0] > 0:
            sample_context = np.array(context_frames[0])  # First context frame
            print("\n  First context frame:")
            print(f"  - Shape: {sample_context.shape}")
            if sample_context.ndim > 2:
                print(
                    f"  - Channel values: {[np.mean(sample_context[..., c]) for c in range(sample_context.shape[-1])]}"
                )

            # Print a small patch of the first context frame
            if sample_context.ndim == 3:
                print(f"  - 5x5 patch of first channel:\n{sample_context[:5, :5, 0]}")
            else:
                print(f"  - 5x5 patch:\n{sample_context[:5, :5]}")

        # Check a sample frame from predictions
        if (
            isinstance(pred_samples, list)
            and len(pred_samples) > 0
            and forecast_length > 0
        ):
            # Handle list type
            if hasattr(pred_samples[0], "__getitem__"):
                try:
                    sample_frame = np.array(pred_samples[0][0])
                    print("\n  First prediction frame (sample 0, time 0):")
                    print(f"  - Shape: {sample_frame.shape}")
                    if sample_frame.ndim > 2:
                        print(
                            f"  - Channel values: {[np.mean(sample_frame[..., c]) for c in range(sample_frame.shape[-1])]}"
                        )

                    # Print a small patch of the first prediction frame
                    if sample_frame.ndim == 3:
                        print(
                            f"  - 5x5 patch of first channel:\n{sample_frame[:5, :5, 0]}"
                        )
                    else:
                        print(f"  - 5x5 patch:\n{sample_frame[:5, :5]}")
                except Exception as e:
                    print(f"  - Error accessing first prediction frame: {e}")
        elif (
            not isinstance(pred_samples, list)
            and len(pred_samples) > 0
            and forecast_length > 0
        ):
            # Handle array type
            sample_frame = np.array(pred_samples[0][0])  # First sample, first frame
            print("\n  First prediction frame (sample 0, time 0):")
            print(f"  - Shape: {sample_frame.shape}")
            if sample_frame.ndim > 2:
                print(
                    f"  - Channel values: {[np.mean(sample_frame[..., c]) for c in range(sample_frame.shape[-1])]}"
                )

            # Print a small patch of the first prediction frame
            if sample_frame.ndim == 3:
                print(f"  - 5x5 patch of first channel:\n{sample_frame[:5, :5, 0]}")
            else:
                print(f"  - 5x5 patch:\n{sample_frame[:5, :5]}")

        print("\n" + "-" * 50)


# Generate videos
print("Generating prediction and statistics videos...")
debug_prediction_samples(all_results)
create_prediction_video(all_results)
create_statistics_video(all_results)

print("All visualizations completed!")

# %%
# Main execution
if __name__ == "__main__":
    print(f"Script completed successfully with master seed {master_seed}.")
    print(
        f"Evaluated {num_trajectories} trajectories with {num_samples_per_trajectory} samples each."
    )
# %%
