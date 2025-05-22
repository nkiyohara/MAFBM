# %%
# Force JAX to use CPU
import os

os.environ["JAX_PLATFORM_NAME"] = "cuda"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import diffrax
from sde.jax.train import build_data_and_model

# %%
# Configuration settings
context_length = 25  # First N steps as observations
forecast_length = 25  # Number of steps to predict
output_dir = "enhanced_visualization_results"

# Number of different input trajectories to visualize
num_trajectories = 5

# Number of samples per visualization type
num_samples = 4  # For 4-row visualizations
num_grid_samples = 9  # For 3x3 grid visualizations

# Visualization trajectory index (dataset index to use for main visualization)
viz_trajectory_idx = 10

# Steps to visualize in grid visualizations
grid_steps = [10, 20]  # 10th and 20th prediction step

# Master random seed for reproducibility
master_seed = 42

# Define the models to analyze (with different KL weights)
models = [
    {"name": "classic-sunset-19", "kl_weight": 0.1},
    # Add more models as needed
    # {"name": "stoic-mountain-18", "kl_weight": 1.0},
]

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the dataset and model configuration
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
def reconstruct_frames(model, params, frames_full, ts_full):
    """
    Reconstruct frames using the encoder-decoder pipeline.

    Args:
        model: The VideoSDE model
        params: Trained parameters
        frames_full: Full sequence of frames
        ts_full: Full sequence of timestamps

    Returns:
        Reconstructed frames
    """
    # Encode the full sequence
    h = model.encoder(params, frames_full)
    w = model.content(params, h)

    # Get posterior distribution and sample
    x0_posterior, h_encoded = model.infer(params, h)

    # Sample from posterior (use mean for consistent reconstruction)
    x0_mean = x0_posterior.mean()

    # Decode using content and latent representations
    # Concatenate content (w) with mean latent state for all timesteps
    w_expanded = w[None, :].repeat(len(frames_full), axis=0)
    x_mean_expanded = x0_mean[None, :].repeat(len(frames_full), axis=0)

    reconstructed_frames = model.decoder(
        params, jnp.concatenate([w_expanded, x_mean_expanded], axis=-1)
    )

    return reconstructed_frames


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
        List of reconstruction and forecast frame sequences
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
def generate_ground_truth_futures(
    data_val, traj_idx, context_length, forecast_length, num_samples, base_seed
):
    """Generate multiple ground truth futures with different noise"""
    gt_futures = []

    for i in range(num_samples):
        # Generate different noise seed for each sample
        noise_seed = base_seed + i * 1000

        # Get trajectory with different noise starting from context_length
        test_sequence = data_val.get_trajectory_with_different_noise(
            index=traj_idx,
            noise_seed=noise_seed,
            divergence_step=context_length,
        )

        # Extract the forecast part
        gt_future = test_sequence[context_length : context_length + forecast_length]
        gt_futures.append(gt_future)

    return gt_futures


# %%
def process_single_model_trajectory(model_info, traj_idx, suffix=""):
    """Process a single model and trajectory for comprehensive visualization"""

    model_name = model_info["name"]
    kl_weight = model_info["kl_weight"]

    print(
        f"Processing model: {model_name} (KL weight = {kl_weight}), trajectory: {traj_idx}"
    )

    # Load model and parameters
    params, model, dt, data_val = load_model_and_params(model_name)

    # Set deterministic key
    key = jax.random.PRNGKey(master_seed + traj_idx * 100)

    # Set sequence length for data
    data_val.seq_len = context_length + forecast_length

    # Get base trajectory
    test_sequence = data_val.get_trajectory_with_different_noise(
        traj_idx, master_seed + traj_idx, context_length
    )

    frames_full = test_sequence
    frames_context = frames_full[:context_length]

    # Create timestamps
    ts_full = jnp.arange(len(frames_full)) * dt * int_sub_steps
    ts_context = ts_full[:context_length]
    ts_forecast = ts_full[context_length : context_length + forecast_length]

    # ===== 1. RECONSTRUCT CONTEXT FRAMES =====
    print("Generating reconstructions...")

    # Generate multiple reconstruction samples
    recon_key, key = jax.random.split(key)
    recon_keys = jax.random.split(recon_key, num_samples)

    reconstructions = []
    for i in range(num_samples):
        # Use the full sequence for reconstruction
        recon_frames = reconstruct_frames(model, params, frames_full, ts_full)
        # Take only the context part
        recon_context = recon_frames[:context_length]
        reconstructions.append(recon_context)

    # ===== 2. GENERATE GROUND TRUTH FUTURES =====
    print("Generating ground truth futures...")

    gt_futures = generate_ground_truth_futures(
        data_val,
        traj_idx,
        context_length,
        forecast_length,
        num_samples,
        master_seed + traj_idx * 100,
    )

    # Generate additional samples for grid visualization
    gt_grid_futures = generate_ground_truth_futures(
        data_val,
        traj_idx,
        context_length,
        forecast_length,
        num_grid_samples,
        master_seed + traj_idx * 200,
    )

    # ===== 3. GENERATE PREDICTIONS (RECURSIVE) =====
    print("Generating recursive predictions...")

    pred_key, key = jax.random.split(key)
    pred_keys = jax.random.split(pred_key, num_samples)

    # JIT-compile forecast function for faster sampling
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

    predictions = []
    for i in range(num_samples):
        forecast_sample = jitted_forecast(pred_keys[i], frames_context)
        predictions.append(forecast_sample)

    # Generate additional predictions for grid visualization
    pred_grid_key, key = jax.random.split(key)
    pred_grid_keys = jax.random.split(pred_grid_key, num_grid_samples)

    grid_predictions = []
    for i in range(num_grid_samples):
        forecast_sample = jitted_forecast(pred_grid_keys[i], frames_context)
        grid_predictions.append(forecast_sample)

    # ===== 4. VISUALIZATIONS =====

    # 4.1 Context and Reconstruction visualization
    print("Creating context and reconstruction visualization...")

    fig, axes = plt.subplots(2, context_length, figsize=(25, 2.1))

    # First row: original context
    for t in range(context_length):
        if frames_context[t].shape[-1] == 1:
            axes[0, t].imshow(frames_context[t, ..., 0], cmap="gray", vmin=0, vmax=1)
        else:
            axes[0, t].imshow(frames_context[t], cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_xticks([])
        axes[0, t].set_yticks([])

    # Second row: reconstruction
    for t in range(context_length):
        if reconstructions[0][t].shape[-1] == 1:
            axes[1, t].imshow(
                reconstructions[0][t, ..., 0], cmap="gray", vmin=0, vmax=1
            )
        else:
            axes[1, t].imshow(reconstructions[0][t], cmap="gray", vmin=0, vmax=1)
        axes[1, t].set_xticks([])
        axes[1, t].set_yticks([])

    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    plt.savefig(
        os.path.join(
            output_dir, f"context_and_reconstructions_{model_name}{suffix}.png"
        ),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # 4.2 Ground truth futures visualization
    print("Creating ground truth futures visualization...")

    fig, axes = plt.subplots(num_samples, forecast_length, figsize=(25, 4))

    for i in range(num_samples):
        for t in range(forecast_length):
            if gt_futures[i][t].shape[-1] == 1:
                axes[i, t].imshow(gt_futures[i][t, ..., 0], cmap="gray", vmin=0, vmax=1)
            else:
                axes[i, t].imshow(gt_futures[i][t], cmap="gray", vmin=0, vmax=1)
            axes[i, t].set_xticks([])
            axes[i, t].set_yticks([])

    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    plt.savefig(
        os.path.join(output_dir, f"ground_truth_futures_{model_name}{suffix}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # 4.3 Predictions visualization
    print("Creating predictions visualization...")

    fig, axes = plt.subplots(num_samples, forecast_length, figsize=(25, 4))

    for i in range(num_samples):
        for t in range(forecast_length):
            if predictions[i][t].shape[-1] == 1:
                axes[i, t].imshow(
                    predictions[i][t, ..., 0], cmap="gray", vmin=0, vmax=1
                )
            else:
                axes[i, t].imshow(predictions[i][t], cmap="gray", vmin=0, vmax=1)
            axes[i, t].set_xticks([])
            axes[i, t].set_yticks([])

    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    plt.savefig(
        os.path.join(output_dir, f"predictions_recursive_{model_name}{suffix}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    return {
        "gt_futures": gt_futures,
        "predictions": predictions,
        "reconstructions": reconstructions,
        "frames_context": frames_context,
        "gt_grid_futures": gt_grid_futures,
        "grid_predictions": grid_predictions,
        "ts_context": ts_context,
        "ts_forecast": ts_forecast,
    }


# %%
def create_grid_visualizations(results_dict, step_idx, model_name, suffix=""):
    """Create 3x3 grid visualizations for specific steps"""
    step = grid_steps[step_idx]

    # Ground truth grid
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    gt_grid_futures = results_dict["gt_grid_futures"]

    for i in range(3):
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            sample_idx = i * 3 + j
            t_step = step - 1  # Convert to 0-indexed

            if gt_grid_futures[sample_idx][t_step].shape[-1] == 1:
                axes[i, j].imshow(
                    gt_grid_futures[sample_idx][t_step, ..., 0],
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
            else:
                axes[i, j].imshow(
                    gt_grid_futures[sample_idx][t_step], cmap="gray", vmin=0, vmax=1
                )

    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    plt.savefig(
        os.path.join(output_dir, f"grid_gt_step{step}_{model_name}{suffix}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # Predictions grid
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    grid_predictions = results_dict["grid_predictions"]

    for i in range(3):
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            sample_idx = i * 3 + j
            t_step = step - 1  # Convert to 0-indexed

            if grid_predictions[sample_idx][t_step].shape[-1] == 1:
                axes[i, j].imshow(
                    grid_predictions[sample_idx][t_step, ..., 0],
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
            else:
                axes[i, j].imshow(
                    grid_predictions[sample_idx][t_step], cmap="gray", vmin=0, vmax=1
                )

    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    plt.savefig(
        os.path.join(
            output_dir, f"grid_predictions_step{step}_{model_name}{suffix}.png"
        ),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


# %%
def visualize_all_models():
    """Main function to visualize all models and trajectories"""

    print(
        f"Starting comprehensive visualization for {len(models)} models and {len(range(num_trajectories))} trajectories..."
    )

    # Process each model
    for model_info in models:
        model_name = model_info["name"]
        print(f"\n=== Processing model: {model_name} ===")

        # Process different trajectories for this model
        all_results = {}

        for i in range(num_trajectories):
            traj_idx = viz_trajectory_idx + i  # Use different trajectory indices
            suffix = f"_traj{traj_idx}"

            print(
                f"\nProcessing trajectory {i + 1}/{num_trajectories} (index: {traj_idx})"
            )

            results = process_single_model_trajectory(model_info, traj_idx, suffix)
            all_results[traj_idx] = results

            # Create grid visualizations for this trajectory
            print("Creating grid visualizations...")
            for step_idx in range(len(grid_steps)):
                create_grid_visualizations(results, step_idx, model_name, suffix)

        print(f"Completed visualization for model: {model_name}")

    print(f"\nAll visualizations completed! Results saved to: {output_dir}")


# %%
def create_comparison_visualizations():
    """Create side-by-side comparison visualizations across models (if multiple models)"""

    if len(models) < 2:
        print("Skipping comparison visualizations (need at least 2 models)")
        return

    print("Creating comparison visualizations...")

    # This would compare predictions from different models on the same trajectory
    # Implementation depends on specific comparison needs

    # Example structure:
    # fig, axes = plt.subplots(len(models), forecast_length, figsize=(25, 2*len(models)))
    # for each model:
    #     for each timestep:
    #         plot prediction

    pass


# %%
# Main execution
if __name__ == "__main__":
    print("Enhanced VideoSDE Visualization")
    print(f"Master seed: {master_seed}")
    print(f"Context length: {context_length}")
    print(f"Forecast length: {forecast_length}")
    print(f"Number of trajectories: {num_trajectories}")
    print(f"Grid steps: {grid_steps}")
    print(f"Models to process: {[m['name'] for m in models]}")

    # Run main visualization
    visualize_all_models()

    # Create comparison visualizations if multiple models
    create_comparison_visualizations()

    print("\n=== Visualization Summary ===")
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    print("- Context and reconstruction comparisons")
    print("- Ground truth future samples")
    print("- Model prediction samples")
    print("- 3x3 grid visualizations for specific timesteps")

    print("\nVisualization completed successfully!")

# %%
