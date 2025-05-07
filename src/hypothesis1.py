import numpy as np
import matplotlib.pyplot as plt
from environmental_factors import EnvironmentalFactors
from scipy.ndimage import gaussian_filter
import random
from scipy.stats import ttest_ind



class WildfireSimulation:
    def __init__(self, grid_size=25, max_iterations=50, convergence_threshold=0.0001):
        self.grid_size = grid_size
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.env = EnvironmentalFactors(grid_size=grid_size)

        # Load static features
        self.elevation = self.env.generate_elevation_grid()
        self.vegetation = self.env.generate_vegetation_ignition_grid(density_factor=3)

        # Load ignition points (fire presence)
        self.ignition_points = np.load('fire_frequent_cells_25x25_southeast_us.npy')

        # Initialize fire spread grid
        self.fire_grid = np.zeros((grid_size, grid_size))
        self.burned_areas = []

        # Store environmental conditions for each iteration
        self.environmental_history = []

    def sample_environmental_conditions(self,high_humidity =False):
        """Sample new environmental conditions for the current iteration."""
        quarter = random.choice(['Q1', 'Q2', 'Q3', 'Q4'])
        humidity = self.env.generate_humidity_grid(quarter, high_humidity= high_humidity)
        wind_speed, wind_direction = self.env.generate_wind_grids(quarter)
        temperature = self.env.temperature_grid(quarter)
        conditions = {
            'humidity': humidity,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'temperature': temperature,
            'quarter': quarter,

        }
        self.environmental_history.append(conditions)
        return conditions

    def wind_influence(self, wind_dir, di, dj):
        # wind_dir in degrees, di/dj are -1, 0, 1
        if di == 0 and dj == 0:
            return 1.0
        cell_angle = np.arctan2(dj, di) * 180 / np.pi
        angle_diff = abs(cell_angle - wind_dir)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        return max(0, np.cos(np.radians(angle_diff)))  # 1 if aligned, 0 if perpendicular, 0 if opposite

    def humidity_factor(self, h):
        return max(0, 1 - h / 100)

    def calculate_spread_probability(self, i, j, conditions, di, dj, ni, nj):
        # Base probability from ignition points and vegetation
        base_prob = self.ignition_points[i, j] * self.vegetation[i, j]
        temp_effect = (conditions['temperature'][i, j] - np.min(conditions['temperature'])) / \
                      (np.max(conditions['temperature']) - np.min(conditions['temperature']))
        humidity_effect = 1 - (conditions['humidity'][i, j] - np.min(conditions['humidity'])) / \
                          (np.max(conditions['humidity']) - np.min(conditions['humidity']))
        wind_effect = (conditions['wind_speed'][i, j] - np.min(conditions['wind_speed'])) / \
                      (np.max(conditions['wind_speed']) - np.min(conditions['wind_speed']))
        elevation_effect = 1 - (self.elevation[i, j] - np.min(self.elevation)) / \
                           (np.max(self.elevation) - np.min(self.elevation))
        weights = {
            'base': 0.2,
            'temperature': 0.45,
            'humidity': 0.4,
            'wind': 0.32,
            'elevation': 0.4,
            'vegetation': 0.3,
        }
        spread_prob = (
                weights['base'] * base_prob +
                weights['temperature'] * temp_effect +
                weights['humidity'] * humidity_effect +
                weights['wind'] * wind_effect +
                weights['elevation'] * elevation_effect
        )
        # Wind and humidity physical influence
        wind_factor = self.wind_influence(conditions['wind_direction'][ni, nj], di, dj)
        humidity_mod = self.humidity_factor(conditions['humidity'][i, j])
        spread_prob *= wind_factor * humidity_mod
        spread_prob = max(spread_prob, 0.25)
        return spread_prob

    def simulate_spread(self,high_humidity=False):
        """Simulate wildfire spread until convergence or max iterations."""

        self.fire_grid = self.ignition_points.copy()
        self.burned_areas = [np.sum(self.fire_grid == 1)]
        k = 30  # Number of consecutive steps for convergence
        recent_changes = []
        for iteration in range(self.max_iterations):
            conditions = self.sample_environmental_conditions(high_humidity=high_humidity)
            new_fire_grid = self.fire_grid.copy()
            spread_probs = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.fire_grid[i, j] == 0:
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                ni, nj = i + di, j + dj
                                if (0 <= ni < self.grid_size and 0 <= nj < self.grid_size and self.fire_grid[
                                    ni, nj] == 1):
                                    spread_prob = self.calculate_spread_probability(i, j, conditions, di, dj, ni, nj)
                                    # if np.random.rand() < 0.05:
                                    #     continue
                                    spread_probs.append(spread_prob)
                                    if np.random.random() < spread_prob:
                                        new_fire_grid[i, j] = 1
            new_fire_grid[self.fire_grid == 1] = 2
            self.fire_grid = new_fire_grid
            burned_area = np.sum(self.fire_grid == 2)
            self.burned_areas.append(burned_area)
            if spread_probs:
                print(f"Iteration {iteration + 1}, avg spread prob: {np.mean(spread_probs):.3f}")
            # Convergence: relative change in burned area <1% for k consecutive steps
            if len(self.burned_areas) > 1:
                rel_change = abs(self.burned_areas[-1] - self.burned_areas[-2]) / max(1, self.burned_areas[-2])
                recent_changes.append(rel_change)
                if len(recent_changes) > k:
                    recent_changes.pop(0)
                # if len(recent_changes) == k and all(change < 0.005 for change in recent_changes):
                #     print(f"Simulation converged after {iteration + 1} iterations (burned area stabilized)")
                #     break
                if len(recent_changes) == k and all(change < 0.01 for change in recent_changes):
                    if np.sum(self.fire_grid == 1) == 0:  # No active fires
                        print(f"Simulation converged after {iteration + 1} iterations (burned area stabilized)")
                        break

        return self.fire_grid, self.burned_areas

    def visualize_results(self):
        """Visualize the simulation results."""
        # Plot final fire spread
        plt.figure(figsize=(10, 8))
        plt.imshow(self.fire_grid, cmap='Reds', origin='lower',
                   extent=[self.env.min_lon, self.env.max_lon,
                           self.env.min_lat, self.env.max_lat],
                   aspect='auto')
        plt.colorbar(label='Fire Spread')
        if self.environmental_history:
            last_quarter = self.environmental_history[-1].get('quarter', 'Unknown')
        else:
            last_quarter = 'Unknown'
        plt.title(f'Wildfire Spread Simulation (Quarter {last_quarter})')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig('wildfire_spread_simulation.png')
        plt.close()

        # Plot burned area over time
        plt.figure(figsize=(10, 4))
        plt.plot(self.burned_areas)
        plt.title('Burned Area Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Burned Area (cells)')
        plt.grid(True)
        plt.savefig('burned_area_over_time.png')
        plt.close()

        # Plot environmental conditions for the last iteration
        if self.environmental_history:
            last_conditions = self.environmental_history[-1]
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Temperature
            im0 = axes[0, 0].imshow(last_conditions['temperature'], cmap='Reds', origin='lower',
                                    extent=[self.env.min_lon, self.env.max_lon,
                                            self.env.min_lat, self.env.max_lat])
            axes[0, 0].set_title('Temperature')
            plt.colorbar(im0, ax=axes[0, 0])

            # Humidity
            im1 = axes[0, 1].imshow(last_conditions['humidity'], cmap='Blues', origin='lower',
                                    extent=[self.env.min_lon, self.env.max_lon,
                                            self.env.min_lat, self.env.max_lat])
            axes[0, 1].set_title('Humidity')
            plt.colorbar(im1, ax=axes[0, 1])

            # Wind Speed
            im2 = axes[1, 0].imshow(last_conditions['wind_speed'], cmap='Greens', origin='lower',
                                    extent=[self.env.min_lon, self.env.max_lon,
                                            self.env.min_lat, self.env.max_lat])
            axes[1, 0].set_title('Wind Speed')
            plt.colorbar(im2, ax=axes[1, 0])

            # Wind Direction
            im3 = axes[1, 1].imshow(last_conditions['wind_direction'], cmap='Purples', origin='lower',
                                    extent=[self.env.min_lon, self.env.max_lon,
                                            self.env.min_lat, self.env.max_lat])
            axes[1, 1].set_title('Wind Direction')
            plt.colorbar(im3, ax=axes[1, 1])

            plt.tight_layout()
            plt.savefig('final_environmental_conditions.png')
            plt.close()


def main():
    # Create and run simulation
    sim = WildfireSimulation()
    fire_grid, burned_areas = sim.simulate_spread()

    # Save results
    np.save('final_fire_spread.npy', fire_grid)
    np.save('burned_areas.npy', np.array(burned_areas))

    # Visualize results
    sim.visualize_results()

    # Adding Convergence plot
    # Load the burned areas data
    burned_areas = np.load('burned_areas.npy')

    # Calculate relative changes
    relative_changes = [
        abs(burned_areas[i] - burned_areas[i - 1]) / max(1, burned_areas[i - 1])
        for i in range(1, len(burned_areas))
    ]

    # Plot the convergence graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(burned_areas)), relative_changes, marker='o', linestyle='-')
    plt.axhline(y=0.01, color='r', linestyle='--', label='Convergence Threshold (1%)')
    plt.title('Convergence Analysis of Burned Area')
    plt.xlabel('Iteration')
    plt.ylabel('Relative Change in Burned Area')
    plt.grid(True)
    plt.legend()
    plt.savefig('convergence_analysis.png')
    plt.show()

    print("Convergence analysis plot saved as 'convergence_analysis.png'")

    print(f"Final burned area: {burned_areas[-1]} cells")
    print(f"Total iterations: {len(burned_areas) - 1}")


if __name__ == "__main__":
    main()

N_RUNS = 100
scenarios = {'baseline': False, 'high_humidity': True}
results = {}


for scenario_name, high_humidity in scenarios.items():
    final_burned_areas = []
    all_iterations = []
    for run in range(N_RUNS):
        sim = WildfireSimulation()
        fire_grid, burned_areas = sim.simulate_spread(high_humidity=high_humidity)
        final_burned_areas.append(burned_areas[-1])
        all_iterations.append(len(burned_areas) - 1)
    results[scenario_name] = {
        'burned_areas': final_burned_areas,
        'iterations': all_iterations,
        'mean_burned': np.mean(final_burned_areas),
        'std_burned': np.std(final_burned_areas),
    }
    print(f"{scenario_name.capitalize()} - Average burned area: {np.mean(final_burned_areas):.2f} ± {np.std(final_burned_areas):.2f}")
    print(f"{scenario_name.capitalize()} - Average iterations: {np.mean(all_iterations):.2f} ± {np.std(all_iterations):.2f}")

baseline_mean = results['baseline']['mean_burned']
high_humidity_mean = results['high_humidity']['mean_burned']
reduction = ((baseline_mean - high_humidity_mean) / baseline_mean) * 100
print(f"\nReduction in burned area due to high humidity: {reduction:.2f}%")

plt.hist(results['baseline']['burned_areas'], bins=20, alpha=0.5, label='Baseline')
plt.hist(results['high_humidity']['burned_areas'], bins=20, alpha=0.5, label='High Humidity')
plt.xlabel('Final Burned Area (cells)')
plt.ylabel('Frequency')
plt.title('Distribution of Burned Areas')
plt.legend()
plt.show()
