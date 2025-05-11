import numpy as np
import matplotlib.pyplot as plt
import random
from environmental_factors import EnvironmentalFactors


class WildfireSimulation:
    def __init__(self, grid_size=25, max_iterations=100, convergence_threshold=0.01, hypothesis_dense_shrub=False):
        self.grid_size = grid_size
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.env = EnvironmentalFactors(grid_size)
        self.elevation = self.env.generate_elevation_grid()
        self.vegetation = self.env.generate_vegetation_ignition_grid(density_factor=3, hypothesis_dense_shrub=hypothesis_dense_shrub)
        self.ignition_points = np.load('fire_frequent_cells_25x25_southeast_us.npy')
        self.fire_grid = np.zeros((grid_size, grid_size))
        self.burned_areas = []
        self.environmental_history = []

    def normalize(self, arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-6)

    def sample_conditions(self):
        q = random.choice(['Q1', 'Q2', 'Q3', 'Q4'])
        conds = {
            'quarter': q,
            'humidity': self.env.generate_humidity_grid(q),
            'wind_speed': self.env.generate_wind_grids(q)[0],
            'wind_direction': self.env.generate_wind_grids(q)[1],
            'temperature': self.env.temperature_grid(q)
        }
        self.environmental_history.append(conds)
        return conds

    def wind_influence(self, wind_dir, di, dj):
        if di == 0 and dj == 0: return 1.0
        angle_diff = abs(np.arctan2(dj, di) * 180 / np.pi - wind_dir) % 360
        if angle_diff > 180: angle_diff = 360 - angle_diff
        return max(0, np.cos(np.radians(angle_diff)))

    def humidity_factor(self, h): return max(0, 1 - h / 100)

    def calc_spread_prob(self, i, j, di, dj, ni, nj, conds):
        base = self.ignition_points[i, j] * self.vegetation[i, j]
        weights = {'base': 0.2, 'temp': 0.3, 'hum': 0.5, 'wind': 0.25, 'elev': 0.35}

        prob = (
            weights['base'] * base +
            weights['temp'] * self.normalize(conds['temperature'])[i, j] +
            weights['hum'] * (1 - self.normalize(conds['humidity'])[i, j]) +
            weights['wind'] * self.normalize(conds['wind_speed'])[i, j] +
            weights['elev'] * (1 - self.normalize(self.elevation)[i, j])
        )
        prob *= self.wind_influence(conds['wind_direction'][ni, nj], di, dj)
        prob *= self.humidity_factor(conds['humidity'][i, j])
        return max(prob, 0.20)

    def simulate_spread(self):
        self.fire_grid = self.ignition_points.copy()
        self.burned_areas = [np.sum(self.fire_grid == 1)]
        recent_changes, k = [], 30

        for iteration in range(self.max_iterations):
            conds = self.sample_conditions()
            new_grid = self.fire_grid.copy()
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.fire_grid[i, j] == 0:
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size and self.fire_grid[ni, nj] == 1:
                                    if np.random.rand() < self.calc_spread_prob(i, j, di, dj, ni, nj, conds):
                                        new_grid[i, j] = 1
            new_grid[self.fire_grid == 1] = 2
            self.fire_grid = new_grid
            self.burned_areas.append(np.sum(new_grid == 2))

            rel_change = abs(self.burned_areas[-1] - self.burned_areas[-2]) / max(1, self.burned_areas[-2])
            recent_changes.append(rel_change)
            if len(recent_changes) > k:
                recent_changes.pop(0)
            if len(recent_changes) == k and all(c < self.convergence_threshold for c in recent_changes):
                if np.sum(self.fire_grid == 1) == 0:
                    break
        return self.fire_grid, self.burned_areas

    def plot_fire_map(self, path='wildfire_spread.png'):
        plt.imshow(self.fire_grid, cmap='Reds', origin='lower', extent=[self.env.min_lon, self.env.max_lon, self.env.min_lat, self.env.max_lat])
        plt.colorbar(label='Fire Spread')
        plt.title(f"Wildfire Spread (Quarter: {self.environmental_history[-1]['quarter']})")
        plt.savefig(path)
        plt.close()

    def plot_burned_area(self, path='burned_area.png'):
        plt.plot(self.burned_areas)
        plt.title('Burned Area Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Burned Cells')
        plt.grid(True)
        plt.savefig(path)
        plt.close()

    def plot_convergence(self, path='convergence_analysis.png'):
        rel_changes = [abs(self.burned_areas[i] - self.burned_areas[i-1]) / max(1, self.burned_areas[i-1]) for i in range(1, len(self.burned_areas))]
        plt.plot(range(1, len(self.burned_areas)), rel_changes, marker='o')
        plt.axhline(y=self.convergence_threshold, color='r', linestyle='--', label='Threshold')
        plt.title('Convergence of Burned Area')
        plt.xlabel('Iteration')
        plt.ylabel('Relative Change')
        plt.grid(True)
        plt.legend()
        plt.savefig(path)
        plt.close()


def moving_average(arr, window):
    """Compute moving average with given window size."""
    return np.convolve(arr, np.ones(window)/window, mode='valid')

def monte_carlo_convergence_check(percentages, window=10, last_n=5, threshold=0.01):
    """Check if the moving average of burned area percentage has converged."""
    ma = moving_average(percentages, window)
    rel_changes = np.abs(np.diff(ma) / (ma[:-1] + 1e-8))
    if len(rel_changes) < last_n:
        return False, ma, rel_changes
    last_changes = rel_changes[-last_n:]
    converged = np.all(last_changes < threshold)
    return converged, ma, rel_changes

def plot_monte_carlo_convergence(percentages, ma, window=10, path='mc_convergence.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, label='Burned Area % per Run', alpha=0.5, color='blue')
    plt.plot(range(window-1, window-1+len(ma)), ma, label=f'Moving Avg (window={window})', color='red', linewidth=2)
    plt.xlabel('Simulation Run')
    plt.ylabel('Burned Area (%)')
    plt.title('Monte Carlo Convergence of Burned Area')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def run_simulation(n_runs=100, mc_window=10, mc_last_n=5, mc_threshold=0.01):
    results = []
    for i in range(n_runs):
        sim = WildfireSimulation()
        _, burned = sim.simulate_spread()
        burned_cells = burned[-1]
        total_cells = sim.grid_size * sim.grid_size
        burned_pct = 100 * burned_cells / total_cells
        results.append((burned_cells, len(burned)-1, burned_pct))
        print(f"Run {i+1}/{n_runs}: Burned {burned_pct:.2f}% of area ({burned_cells} cells) in {len(burned)-1} iterations")
    
    areas, iterations, percentages = zip(*results)
    print("\nSummary Statistics:")
    print(f"Average burned area: {np.mean(areas):.2f} ± {np.std(areas):.2f}")
    print(f"Average iterations: {np.mean(iterations):.2f} ± {np.std(iterations):.2f}")
    print(f"Average burned area %: {np.mean(percentages):.2f} ± {np.std(percentages):.2f}")

    # Monte Carlo convergence check
    converged, ma, rel_changes = monte_carlo_convergence_check(percentages, window=mc_window, last_n=mc_last_n, threshold=mc_threshold)
    if converged:
        print(f"\nMonte Carlo convergence achieved: relative change < {mc_threshold*100:.2f}% for last {mc_last_n} steps.")
    else:
        print("\nMonte Carlo convergence NOT achieved.")

    # Plot histogram of burned areas
    plt.figure(figsize=(10, 6))
    plt.hist(percentages, bins=20, color='orange', edgecolor='k')
    plt.xlabel('Final Burned Area (%)')
    plt.ylabel('Frequency')
    plt.title('Burned Area Percentage Distribution')
    plt.grid(True)
    plt.savefig('burned_area_distribution.png')
    plt.close()

    # Plot MC convergence
    plot_monte_carlo_convergence(percentages, ma, window=mc_window)


if __name__ == "__main__":
    run_simulation(n_runs=1000)  # Run 1000 simulations for better convergence analysis
