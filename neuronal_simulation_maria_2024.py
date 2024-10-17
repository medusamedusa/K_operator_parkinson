
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Parameters
num_neurons = 100  # Number of neurons
sim_time = 1000  # Simulation time in ms
tstep = 1  # Time step in ms
tau_m = 20  # Membrane time constant in ms
v_rest = 0  # Resting potential in mV
v_reset = 0  # Reset potential in mV
v_th = 1  # Threshold potential for spiking in mV
refractory_period = 5  # Refractory period in ms
syn_strength = 0.05  # Synaptic strength multiplier

# Healthy persons
#interaction_densities = [0.7, 0.8, 0.5, 0.5]

# long Covid
#interaction_densities = [0.4, 0.3, 0.8, 0.4]

# PD
interaction_densities = [0.4, 0.3, 0.2, 0.8]

# Function to generate synaptic weights matrix
def generate_weights(num_neurons, interaction_density):
    weights = np.random.rand(num_neurons, num_neurons) * interaction_density
    sparse_mask = np.random.rand(num_neurons, num_neurons) < interaction_density
    weights *= sparse_mask
    np.fill_diagonal(weights, 0)  # No self-connections
    return weights

# LIF Neuron simulation
def simulate_lif_network(num_neurons, weights_matrix, sim_time, tstep, tau_m, v_th, v_reset, v_rest, syn_strength):
    num_steps = int(sim_time / tstep)
    V = np.ones(num_neurons) * v_rest  # Initialize all neurons at resting potential
    spikes = np.zeros((num_neurons, num_steps))  # To record spikes
    refractory_time = np.zeros(num_neurons)  # Track refractory period for neurons

    for t in range(1, num_steps):
        # Update all neurons
        for i in range(num_neurons):
            if refractory_time[i] > 0:
                refractory_time[i] -= tstep  # Neuron is in refractory period
                V[i] = v_reset
            else:
                # Synaptic input from connected neurons
                I_syn = np.dot(weights_matrix[i, :], spikes[:, t-1]) * syn_strength
                # Update membrane potential with Euler's method
                dV = (-V[i] + v_rest + I_syn) * (tstep / tau_m)
                V[i] += dV

                # Check if neuron fires
                if V[i] >= v_th:
                    spikes[i, t] = 1  # Record spike
                    V[i] = v_reset  # Reset membrane potential
                    refractory_time[i] = refractory_period  # Apply refractory period
    
    return spikes

# Custom colormap: white for 0, blue for -1, and green for +1
def create_custom_colormap():
    colors = [(1, 1, 1), (0, 0, 1), (0, 1, 0)]  # White, Blue, Green
    n_bins = 100  # Number of bins for interpolation
    cmap_name = 'custom_cmap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return cm

# Set up the figure for plotting 4 panels (rectangular heatmaps)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Fixed color scale between -1 and 1 for all heatmaps
vmin, vmax = -1, 1
custom_cmap = create_custom_colormap()

# Run the simulation for each density and plot results
for idx, interaction_density in enumerate(interaction_densities):
    # Generate synaptic weights matrix for current density
    weights_matrix = generate_weights(num_neurons, interaction_density)
    
    # Run the LIF simulation
    spikes = simulate_lif_network(num_neurons, weights_matrix, sim_time, tstep, tau_m, v_th, v_reset, v_rest, syn_strength)
    
    # Plot the synaptic weights heatmap
    ax = axes[idx]
    sns.heatmap(weights_matrix, cmap=custom_cmap, cbar=True, ax=ax, vmin=vmin, vmax=vmax)
    
    # Update titles with the correct labels
    if idx == 0:
        ax.set_title(f"Synaptic Weights, ROI 161 (Density {interaction_density})")
    elif idx == 1:
        ax.set_title(f"Synaptic Weights, ROI 162 (Density {interaction_density})")
    elif idx == 2:
        ax.set_title(f"Synaptic Weights, ROI 150 (Density {interaction_density})")
    elif idx == 3:
        ax.set_title(f"Synaptic Weights, ROI 38 (Density {interaction_density})")
    
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Neuron Index')

    # Calculate and print the average value of the synaptic weights matrix
    avg_value = np.mean(weights_matrix)
    print(f"Average synaptic weight for density {interaction_density}: {avg_value:.4f}")

# Adjust the layout and show the plot
plt.tight_layout()

# Export the figure as a PDF
#plt.savefig('synaptic_weights_heatmaps_healthy.pdf', format='pdf')
#plt.savefig('synaptic_weights_heatmaps_Covid.pdf', format='pdf')
plt.savefig('synaptic_weights_heatmaps_PD.pdf', format='pdf')


plt.show()
