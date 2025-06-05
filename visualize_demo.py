#!/usr/bin/env python3
"""
Universal Physics Transformer (UPT) Demo Visualization
Shows the physics simulation results from the trained model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append('src')

def load_rollout_data():
    """Load the saved rollout results from training"""
    rollout_path = Path("save/stage1/6a54alle/rollout/rollout_results_e2_u6_s48.pt")
    if not rollout_path.exists():
        # Try to find the most recent rollout
        rollout_dir = Path("save/stage1")
        if rollout_dir.exists():
            for subdir in rollout_dir.iterdir():
                if subdir.is_dir():
                    rollout_files = list((subdir / "rollout").glob("*.pt"))
                    if rollout_files:
                        rollout_path = rollout_files[-1]  # Take the last one
                        break
    
    print(f"Loading rollout data from: {rollout_path}")
    return torch.load(rollout_path)

def visualize_kinetic_energy(rollout_data):
    """Plot kinetic energy conservation over time"""
    ekin_target = rollout_data['ekin_target'].cpu().numpy()
    ekin_pred = rollout_data['ekin_predictions'].cpu().numpy()
    time_idx = rollout_data['time_idx'].cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    
    # Plot mean kinetic energy over all trajectories
    plt.subplot(1, 2, 1)
    plt.plot(time_idx, ekin_target.mean(axis=0), 'b-', label='Target', linewidth=2)
    plt.plot(time_idx, ekin_pred.mean(axis=0), 'r--', label='Predicted', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Kinetic Energy')
    plt.title('Kinetic Energy Conservation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot error over time
    plt.subplot(1, 2, 2)
    error = np.abs(ekin_pred - ekin_target).mean(axis=0)
    plt.plot(time_idx, error, 'g-', linewidth=2)
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Error')
    plt.title('Kinetic Energy Error')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kinetic_energy_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_particle_animation(rollout_data, traj_idx=0, save_gif=True):
    """Create an animation showing particle motion"""
    vel_target = rollout_data['vel_target'][traj_idx].cpu().numpy()  # [2500, 24, 2]
    vel_pred = rollout_data['vel_predictions'][traj_idx].cpu().numpy()  # [2500, 24, 2]
    
    # Calculate positions by integrating velocities (assuming unit time steps)
    dt = 0.01  # time step
    n_particles, n_timesteps, n_dims = vel_target.shape
    
    # Initialize positions (assuming they start at a grid)
    grid_size = int(np.sqrt(n_particles))
    if grid_size * grid_size != n_particles:
        grid_size = 50  # fallback
        indices = np.random.choice(n_particles, grid_size*grid_size, replace=False)
        vel_target = vel_target[indices]
        vel_pred = vel_pred[indices]
        n_particles = grid_size * grid_size
    
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    pos_init = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    # Integrate to get positions
    pos_target = np.zeros((n_particles, n_timesteps, 2))
    pos_pred = np.zeros((n_particles, n_timesteps, 2))
    
    pos_target[:, 0] = pos_init
    pos_pred[:, 0] = pos_init
    
    for t in range(1, n_timesteps):
        pos_target[:, t] = pos_target[:, t-1] + vel_target[:, t-1] * dt
        pos_pred[:, t] = pos_pred[:, t-1] + vel_pred[:, t-1] * dt
    
    # Create animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Determine plot limits
    all_pos = np.concatenate([pos_target, pos_pred], axis=0)
    xlim = [all_pos[:, :, 0].min() - 0.1, all_pos[:, :, 0].max() + 0.1]
    ylim = [all_pos[:, :, 1].min() - 0.1, all_pos[:, :, 1].max() + 0.1]
    
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_title('Target Particle Motion')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_title('Predicted Particle Motion')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    
    # Sample subset of particles for visualization
    n_viz = min(400, n_particles)
    indices = np.linspace(0, n_particles-1, n_viz).astype(int)
    
    scat1 = ax1.scatter([], [], c=[], s=20, cmap='viridis', alpha=0.7)
    scat2 = ax2.scatter([], [], c=[], s=20, cmap='viridis', alpha=0.7)
    
    def animate(frame):
        # Color by velocity magnitude
        vel_mag_target = np.linalg.norm(vel_target[indices, frame], axis=1)
        vel_mag_pred = np.linalg.norm(vel_pred[indices, frame], axis=1)
        
        scat1.set_offsets(pos_target[indices, frame])
        scat1.set_array(vel_mag_target)
        
        scat2.set_offsets(pos_pred[indices, frame])
        scat2.set_array(vel_mag_pred)
        
        fig.suptitle(f'Particle Dynamics - Time Step: {frame}/{n_timesteps-1}', fontsize=14)
        
        return scat1, scat2
    
    anim = animation.FuncAnimation(fig, animate, frames=n_timesteps, interval=200, blit=False)
    
    if save_gif:
        print("Saving animation as particle_motion.gif...")
        anim.save('particle_motion.gif', writer='pillow', fps=5)
        print("Animation saved!")
    
    plt.tight_layout()
    plt.show()
    
    return anim

def print_simulation_stats(rollout_data):
    """Print statistics about the simulation"""
    ekin_target = rollout_data['ekin_target'].cpu().numpy()
    ekin_pred = rollout_data['ekin_predictions'].cpu().numpy()
    vel_target = rollout_data['vel_target'].cpu().numpy()
    vel_pred = rollout_data['vel_predictions'].cpu().numpy()
    
    print("=" * 60)
    print("UPT PHYSICS SIMULATION DEMO RESULTS")
    print("=" * 60)
    print(f"Number of trajectories: {ekin_target.shape[0]}")
    print(f"Number of particles per trajectory: {vel_target.shape[1]}")
    print(f"Number of time steps: {ekin_target.shape[1]}")
    print(f"Spatial dimensions: {vel_target.shape[3]}")
    print()
    
    # Kinetic energy statistics
    ke_error = np.abs(ekin_pred - ekin_target)
    print("KINETIC ENERGY CONSERVATION:")
    print(f"  Mean absolute error: {ke_error.mean():.6f}")
    print(f"  Max absolute error: {ke_error.max():.6f}")
    print(f"  Relative error: {(ke_error / ekin_target).mean():.3%}")
    print()
    
    # Velocity statistics
    vel_error = np.linalg.norm(vel_pred - vel_target, axis=-1)
    print("VELOCITY PREDICTION:")
    print(f"  Mean velocity error: {vel_error.mean():.6f}")
    print(f"  Max velocity error: {vel_error.max():.6f}")
    print()
    
    # Conservation metrics
    initial_ke = ekin_target[:, 0].mean()
    final_ke_target = ekin_target[:, -1].mean()
    final_ke_pred = ekin_pred[:, -1].mean()
    
    print("ENERGY CONSERVATION:")
    print(f"  Initial kinetic energy: {initial_ke:.6f}")
    print(f"  Final kinetic energy (target): {final_ke_target:.6f}")
    print(f"  Final kinetic energy (predicted): {final_ke_pred:.6f}")
    print(f"  Energy drift (target): {abs(final_ke_target - initial_ke)/initial_ke:.3%}")
    print(f"  Energy drift (predicted): {abs(final_ke_pred - initial_ke)/initial_ke:.3%}")
    print("=" * 60)

def main():
    """Main demo function"""
    print("Universal Physics Transformer (UPT) - Physics Simulation Demo")
    print("Loading rollout data...")
    
    try:
        rollout_data = load_rollout_data()
        print("Rollout data loaded successfully!")
        
        # Print simulation statistics
        print_simulation_stats(rollout_data)
        
        # Create kinetic energy plots
        print("\nGenerating kinetic energy conservation plots...")
        visualize_kinetic_energy(rollout_data)
        
        # Create particle motion animation
        print("\nGenerating particle motion animation...")
        anim = create_particle_animation(rollout_data, traj_idx=0, save_gif=True)
        
        print("\nDemo complete! Check the generated files:")
        print("  - kinetic_energy_results.png: Energy conservation plots")
        print("  - particle_motion.gif: Animated particle dynamics")
        
    except Exception as e:
        print(f"Error running demo: {e}")
        print("Make sure you have run the training first to generate rollout data.")

if __name__ == "__main__":
    main()
