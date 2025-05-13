import numpy as np
import matplotlib.pyplot as plt
import random
from environmental_factors import EnvironmentalFactors
from multiprocessing import Pool, cpu_count
from numba import jit, float64, int64, boolean
from scipy.stats import norm

@jit(nopython=True)
def calculate_spread_probabilities(
    grid_size, burning_cells, fire_grid, ignition_points, vegetation,
    norm_temp, norm_humidity, norm_wind_speed, norm_elevation,
    wind_direction, humidity, weights
):
    """Calculate spread probabilities using Numba JIT compilation."""
    spread_prob = np.zeros((grid_size, grid_size), dtype=np.float64)
    
    for i in range(grid_size):
        for j in range(grid_size):
            if burning_cells[i, j]:
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < grid_size and 
                            0 <= nj < grid_size and 
                            fire_grid[ni, nj] == 0):
                            
                            # Calculate base probability
                            base = ignition_points[ni, nj] * vegetation[ni, nj]
                            
                            prob = (
                                weights[0] * base +
                                weights[1] * norm_temp[ni, nj] +
                                weights[2] * (1 - norm_humidity[ni, nj]) +
                                weights[3] * norm_wind_speed[ni, nj] +
                                weights[4] * (1 - norm_elevation[ni, nj])
                            )
                            
                            # Calculate wind influence
                            if di == 0 and dj == 0:
                                wind_influence = 1.0
                            else:
                                angle_diff = abs(np.arctan2(dj, di) * 180 / np.pi - wind_direction[ni, nj]) % 360
                                if angle_diff > 180:
                                    angle_diff = 360 - angle_diff
                                wind_influence = max(0, np.cos(np.radians(angle_diff)))
                            
                            # Apply wind influence and humidity factor
                            prob *= wind_influence
                            prob *= max(0, 1 - humidity[ni, nj] / 100)
                            spread_prob[ni, nj] = max(prob, 0.20)
    
    return spread_prob

class WildfireSimulation:
    def __init__(self, grid_size=25, max_time_steps=24, convergence_threshold=0.01, hypothesis_dense_shrub=False):
        """
        Initialize wildfire simulation.
        max_time_steps: Maximum number of time steps (e.g., hours) to simulate
        """
        self.grid_size = grid_size
        self.max_time_steps = max_time_steps
        self.convergence_threshold = convergence_threshold
        self.env = EnvironmentalFactors(grid_size)
        
        # Generate static grids once
        self.elevation = self.env.generate_elevation_grid()
        self.vegetation = self.env.generate_vegetation_ignition_grid(density_factor=3, hypothesis_dense_shrub=hypothesis_dense_shrub)
        self.ignition_points = np.load('fire_frequent_cells_25x25_southeast_us.npy')
        
        # Initialize fire grid
        self.fire_grid = np.zeros((grid_size, grid_size))
        self.burned_areas = []
        
        # Generate initial environmental conditions
        self.quarter = random.choice(['Q1', 'Q2', 'Q3', 'Q4'])
        self.humidity = self.env.generate_humidity_grid(self.quarter)
        self.wind_speed, self.wind_direction = self.env.generate_wind_grids(self.quarter)
        self.temperature = self.env.temperature_grid(self.quarter)
        
        # Autoregressive parameters for weather evolution
        self.weather_persistence = 0.8  # How much weather persists between time steps
        self.weather_variability = 0.2  # How much weather can change between time steps

    def normalize(self, arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-6)

    def evolve_weather(self):
        """Evolve weather conditions using a simple autoregressive model."""
        # Add some random variation while maintaining persistence
        self.humidity = (self.weather_persistence * self.humidity + 
                        self.weather_variability * np.random.normal(50, 10, self.humidity.shape))
        self.humidity = np.clip(self.humidity, 0, 100)
        
        self.temperature = (self.weather_persistence * self.temperature + 
                          self.weather_variability * np.random.normal(25, 5, self.temperature.shape))
        
        # Wind direction changes more slowly
        self.wind_direction = (self.weather_persistence * self.wind_direction + 
                             self.weather_variability * np.random.normal(0, 30, self.wind_direction.shape))
        self.wind_direction = self.wind_direction % 360
        
        self.wind_speed = (self.weather_persistence * self.wind_speed + 
                          self.weather_variability * np.random.normal(10, 3, self.wind_speed.shape))
        self.wind_speed = np.clip(self.wind_speed, 0, None)

    def simulate_spread(self):
        """Simulate fire spread over time."""
        self.fire_grid = self.ignition_points.copy()
        self.burned_areas = [np.sum(self.fire_grid == 1)]
        recent_changes, k = [], 5  # Check stopping over last 5 time steps

        for time_step in range(self.max_time_steps):
            # Evolve weather conditions
            self.evolve_weather()
            
            # Calculate spread probabilities
            spread_prob = calculate_spread_probabilities(
                self.grid_size, (self.fire_grid == 1), self.fire_grid,
                self.ignition_points, self.vegetation,
                self.normalize(self.temperature),
                self.normalize(self.humidity),
                self.normalize(self.wind_speed),
                self.normalize(self.elevation),
                self.wind_direction, self.humidity,
                np.array([0.2, 0.3, 0.5, 0.25, 0.35])  # weights
            )
            
            # Apply spread probabilities
            new_fires = (np.random.random(self.fire_grid.shape) < spread_prob) & (self.fire_grid == 0)
            self.fire_grid[new_fires] = 1
            self.fire_grid[self.fire_grid == 1] = 2  # Mark old burning cells as burned
            
            # Update burned area
            burned_area = np.sum(self.fire_grid == 2)
            self.burned_areas.append(burned_area)
            
            # Check convergence
            rel_change = abs(self.burned_areas[-1] - self.burned_areas[-2]) / max(1, self.burned_areas[-2])
            recent_changes.append(rel_change)
            if len(recent_changes) > k:
                recent_changes.pop(0)
            if len(recent_changes) == k and all(c < self.convergence_threshold for c in recent_changes):
                break
                
        return self.fire_grid, self.burned_areas

    def plot_fire_map(self, path='wildfire_spread.png'):
        plt.imshow(self.fire_grid, cmap='Reds', origin='lower', extent=[self.env.min_lon, self.env.max_lon, self.env.min_lat, self.env.max_lat])
        plt.colorbar(label='Fire Spread')
        plt.title(f"Wildfire Spread (Quarter: {self.quarter})")
        plt.savefig(path)
        plt.close()

    def plot_burned_area(self, path='burned_area.png'):
        plt.plot(self.burned_areas)
        plt.title('Burned Area Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Burned Cells')
        plt.grid(True)
        plt.savefig(path)
        plt.close()

    def plot_convergence(self, path='convergence_analysis.png'):
        rel_changes = [abs(self.burned_areas[i] - self.burned_areas[i-1]) / max(1, self.burned_areas[i-1]) for i in range(1, len(self.burned_areas))]
        plt.plot(range(1, len(self.burned_areas)), rel_changes, marker='o')
        plt.axhline(y=self.convergence_threshold, color='r', linestyle='--', label='Threshold')
        plt.title('Convergence of Burned Area')
        plt.xlabel('Time Step')
        plt.ylabel('Relative Change')
        plt.grid(True)
        plt.legend()
        plt.savefig(path)
        plt.close()


def moving_average(arr, window):
    """Compute moving average with given window size."""
    return np.convolve(arr, np.ones(window)/window, mode='valid')

def monte_carlo_convergence_check(percentages, window=20, last_n=10, threshold=0.05):
    """Check if the moving average of burned area percentage has converged.
    Using a larger window and more lenient threshold for better convergence."""
    ma = moving_average(percentages, window)
    rel_changes = np.abs(np.diff(ma) / (ma[:-1] + 1e-8))
    if len(rel_changes) < last_n:
        return False, ma, rel_changes
    last_changes = rel_changes[-last_n:]
    converged = np.all(last_changes < threshold)
    return converged, ma, rel_changes

def plot_monte_carlo_convergence(percentages, ma, window=20, path='mc_convergence.png'):
    plt.figure(figsize=(12, 8))
    
    # Plot individual runs
    plt.subplot(2, 1, 1)
    plt.plot(percentages, label='Burned Area % per Run', alpha=0.3, color='blue')
    plt.plot(range(window-1, window-1+len(ma)), ma, label=f'Moving Avg (window={window})', color='red', linewidth=2)
    plt.xlabel('Simulation Run')
    plt.ylabel('Burned Area (%)')
    plt.title('Monte Carlo Convergence of Burned Area')
    plt.legend()
    plt.grid(True)
    
    # Plot relative changes
    plt.subplot(2, 1, 2)
    rel_changes = np.abs(np.diff(ma) / (ma[:-1] + 1e-8))
    plt.plot(range(window, window+len(rel_changes)), rel_changes, label='Relative Changes', color='green')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Convergence Threshold (5%)')
    plt.xlabel('Simulation Run')
    plt.ylabel('Relative Change')
    plt.title('Relative Changes in Moving Average')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def run_single_simulation(args):
    """Run a single simulation and return its results."""
    sim = WildfireSimulation()
    _, burned = sim.simulate_spread()
    burned_cells = burned[-1]
    total_cells = sim.grid_size * sim.grid_size
    burned_pct = 100 * burned_cells / total_cells
    return (burned_cells, len(burned)-1, burned_pct)

def run_simulation(n_runs=2000, mc_window=20, mc_last_n=10, mc_threshold=0.05):
    # Use number of CPU cores minus 1 to leave one core free
    n_cores = max(1, cpu_count() - 1)
    print(f"Running {n_runs} simulations using {n_cores} CPU cores...")
    
    # Create a pool of workers
    with Pool(n_cores) as pool:
        # Run simulations in parallel
        results = pool.map(run_single_simulation, [None] * n_runs)
    
    areas, iterations, percentages = zip(*results)
    
    # Calculate more detailed statistics
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    mean_iter = np.mean(iterations)
    std_iter = np.std(iterations)
    mean_pct = np.mean(percentages)
    std_pct = np.std(percentages)
    
    print("\nDetailed Summary Statistics:")
    print(f"Burned Area: {mean_area:.2f} ± {std_area:.2f} cells")
    print(f"Iterations: {mean_iter:.2f} ± {std_iter:.2f} steps")
    print(f"Burned Area %: {mean_pct:.2f} ± {std_pct:.2f}%")
    print(f"95% Confidence Interval: [{mean_pct - 1.96*std_pct:.2f}%, {mean_pct + 1.96*std_pct:.2f}%]")

    # Monte Carlo convergence check
    converged, ma, rel_changes = monte_carlo_convergence_check(
        percentages, 
        window=mc_window, 
        last_n=mc_last_n, 
        threshold=mc_threshold
    )
    
    if converged:
        print(f"\nMonte Carlo convergence achieved:")
        print(f"- Relative change < {mc_threshold*100:.2f}% for last {mc_last_n} steps")
        print(f"- Final moving average: {ma[-1]:.2f}%")
        print(f"- Final relative change: {rel_changes[-1]*100:.2f}%")
    else:
        print("\nMonte Carlo convergence NOT achieved.")
        print(f"Last {mc_last_n} relative changes:")
        for i, change in enumerate(rel_changes[-mc_last_n:]):
            print(f"  Step {len(rel_changes)-mc_last_n+i+1}: {change*100:.2f}%")

    # Plot histogram of burned areas with normal distribution fit
    plt.figure(figsize=(12, 6))
    plt.hist(percentages, bins=30, color='orange', edgecolor='k', density=True, alpha=0.7)
    
    # Add normal distribution fit
    x = np.linspace(min(percentages), max(percentages), 100)
    plt.plot(x, norm.pdf(x, mean_pct, std_pct), 'r-', lw=2, label='Normal Distribution Fit')
    
    plt.xlabel('Final Burned Area (%)')
    plt.ylabel('Density')
    plt.title('Burned Area Percentage Distribution with Normal Fit')
    plt.grid(True)
    plt.legend()
    plt.savefig('burned_area_distribution.png')
    plt.close()

    # Plot MC convergence
    plot_monte_carlo_convergence(percentages, ma, window=mc_window)


if __name__ == "__main__":
    run_simulation(n_runs=2000)  # Increased number of runs for better convergence
