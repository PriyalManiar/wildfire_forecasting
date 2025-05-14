import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from final_environmental_factors import EnvironmentalFactors
from multiprocessing import Pool, cpu_count
from numba import jit, float64, int64, boolean
from scipy.stats import norm, ttest_ind


@jit(nopython=True)
def calculate_spread_probabilities(
        grid_size:int, burning_cells:np.ndarray, fire_grid:np.ndarray, ignition_points: np.ndarray, vegetation:np.ndarray,
        norm_temp: np.ndarray, norm_humidity : np.ndarray, norm_wind_speed: np.ndarray, norm_elevation: np.ndarray,
        wind_direction: np.ndarray, humidity: np.ndarray, weights: np.ndarray
)->np.ndarray:

    """Calculation of spread probabilities for each cell in the grid

    :param grid_size: size of the grid
    :param burning_cells: grid of the burning cells
    :param fire_grid: grid showing fire status
    :param ignition_points: initial ignition points
    :param vegetation: Vegetation ignition probabilities
    :param norm_temp: Normalization temperature grid
    :param norm_humidity: Normalized humidity grid
    :param norm_wind_speed: Normalized wind speed grid
    :param norm_elevation: Normalized elevation grid
    :param wind_direction: Wind direction grid (degrees)
    :param humidity: Absolute humidity values
    :param weights: Weight for each environmental factor
    :return np.ndarray: spread probabilities for each cell in the grid

    """
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
                            spread_prob[ni, nj] = max(prob, 0.01)

    return spread_prob


class WildfireSimulation:
    def __init__(self, grid_size:int=25, max_time_steps:int=24, convergence_threshold:int=0.01, hypothesis: [str]=False)->None:
        """
        Initialize wildfire simulation class.

        :param grid_size: size of the simulation grid
        :param max_time_steps: maximum number of time steps for simulation
        :param convergence_threshold: convergence threshold check value for simulation
        :param hypothesis: hypothesis check value for simulation

        """
        self.grid_size = grid_size
        self.max_time_steps = max_time_steps
        self.convergence_threshold = convergence_threshold
        self.hypothesis = hypothesis
        self.env = EnvironmentalFactors(grid_size)

        # Generate static grids once
        self.elevation = self.env.generate_elevation_grid()
        dense_shrub_flag = True if self.hypothesis == 'dense_shrub' else False
        shrub_factor = 1.5 if self.hypothesis == 'dense_shrub' else 1
        self.vegetation = self.env.generate_vegetation_ignition_grid(density_factor=3,
                                                                     hypothesis_dense_shrub=dense_shrub_flag)
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

    def normalize(self, arr:np,ndarray)->np.ndarray:
        """
        Normalise array values between 0 and 1
        :param arr: Input array
        :return: Normalized array

        """
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-6)

    def evolve_weather(self)-> None:

        """
        Evolve environmental conditions for the cells in the grid using autoregressive model.
        """
        # Add some random variation while maintaining persistence
        self.humidity = (self.weather_persistence * self.humidity +
                         self.weather_variability * np.random.normal(50, 10, self.humidity.shape))
        self.humidity = np.clip(self.humidity, 0, 100)

        if self.hypothesis == 'high_humidity':
            self.humidity = np.maximum(self.humidity, 90)

        self.temperature = (self.weather_persistence * self.temperature +
                            self.weather_variability * np.random.normal(25, 5, self.temperature.shape))

        # Wind direction changes more slowly
        self.wind_direction = (self.weather_persistence * self.wind_direction +
                               self.weather_variability * np.random.normal(0, 30, self.wind_direction.shape))
        self.wind_direction = self.wind_direction % 360

        self.wind_speed = (self.weather_persistence * self.wind_speed +
                           self.weather_variability * np.random.normal(10, 3, self.wind_speed.shape))
        self.wind_speed = np.clip(self.wind_speed, 0, None)

        if self.hypothesis == 'high_wind':
            self.wind_speed *= 5.0

    def simulate_spread(self)-> tuple[np.ndarray, list[float]]:

        """
        This function runs the wildfire spread simulation.
        :return: final fire grid and burned area history across various time steps in the grid.
        """

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
                np.array([0.2, 0.3, 1.0, 0.7, 0.35])  # weights
            )

            # Apply spread probabilities
            new_fires = (np.random.random(self.fire_grid.shape) < spread_prob) & (self.fire_grid == 0)
            self.fire_grid[new_fires] = 1
            self.fire_grid[self.fire_grid == 1] = 2  # Mark old burning cells as burned

            # Update burned area
            burned_area = np.sum(self.fire_grid == 2)
            self.burned_areas.append(burned_area)

            if not self.hypothesis:
                rel_change = abs(self.burned_areas[-1] - self.burned_areas[-2]) / max(1, self.burned_areas[-2])
                recent_changes.append(rel_change)
                if len(recent_changes) > k:
                    recent_changes.pop(0)
                if len(recent_changes) == k and all(c < self.convergence_threshold for c in recent_changes):
                    break

            # # Check convergence
            # rel_change = abs(self.burned_areas[-1] - self.burned_areas[-2]) / max(1, self.burned_areas[-2])
            # recent_changes.append(rel_change)
            # if len(recent_changes) > k:
            #     recent_changes.pop(0)
            # if len(recent_changes) == k and all(c < self.convergence_threshold for c in recent_changes):
            #     break
            #
        return self.fire_grid, self.burned_areas

    def plot_fire_map(self, path: str='wildfire_spread.png')-> None:

        """
        This function plots and saves the wildfire spread map
        :param path: file path to save the plot created
        """

        plt.imshow(self.fire_grid, cmap='Reds', origin='lower',
                   extent=[self.env.min_lon, self.env.max_lon, self.env.min_lat, self.env.max_lat])
        plt.colorbar(label='Fire Spread')
        plt.title(f"Wildfire Spread (Quarter: {self.quarter})")
        plt.savefig(path)
        plt.close()

    def plot_burned_area(self, path: str='burned_area.png')-> None:
        """
        This function plots and saves the burned area map over time.
        :param path: file path to save the plot created
        """
        plt.plot(self.burned_areas)
        plt.title('Burned Area Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Burned Cells')
        plt.grid(True)
        plt.savefig(path)
        plt.close()

    def plot_convergence(self, path: str='convergence_analysis.png')-> None:
        """
        This function plots and saves the convergence analysis of the simulation.
        :param path: file path to save the plot created
        """
        rel_changes = [abs(self.burned_areas[i] - self.burned_areas[i - 1]) / max(1, self.burned_areas[i - 1]) for i in
                       range(1, len(self.burned_areas))]
        plt.plot(range(1, len(self.burned_areas)), rel_changes, marker='o')
        plt.axhline(y=self.convergence_threshold, color='r', linestyle='--', label='Threshold')
        plt.title('Convergence of Burned Area')
        plt.xlabel('Time Step')
        plt.ylabel('Relative Change')
        plt.grid(True)
        plt.legend()
        plt.savefig(path)
        plt.close()


def moving_average(arr: list[float], window: int)-> np.ndarray:
    """
    This function returns the moving average over the specified window size.
    :param arr: Input data series
    :param window: Window size for the moving average
    :return: Moving average value for the input data series
    """
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def monte_carlo_convergence_check(percentages: list[float], window: int=20, last_n: int=10, threshold: float=0.05)-> tuple[bool,np.ndarray, np.ndarray]:
    """
    This function checks the Monte Carlo simulation for convergence using relative changes.
    :param percentages: Burned area percentage across the various runs
    :param window: window size for the Monte Carlo simulation moving average
    :param last_n: number of last steps to check for convergence
    :param threshold: convergence threshold value
    :return: moving average and relative changes for the simulation run
    """
    ma = moving_average(percentages, window)
    rel_changes = np.abs(np.diff(ma) / (ma[:-1] + 1e-8))
    if len(rel_changes) < last_n:
        return False, ma, rel_changes
    last_changes = rel_changes[-last_n:]
    converged = np.all(last_changes < threshold)
    return converged, ma, rel_changes


def plot_monte_carlo_convergence(percentages: list[float], ma:np.ndarray, window: int=20, path: str='mc_convergence.png')-> None:
    """
    This function plots the Monte Carlo simulation convergence graph.
    :param percentages: Burned area percentages per run
    :param ma: Moving average values
    :param window: window size used
    :param path: output file path
    """
    plt.figure(figsize=(12, 8))
    # Plot individual runs
    plt.subplot(2, 1, 1)
    plt.plot(percentages, label='Burned Area % per Run', alpha=0.3, color='blue')
    plt.plot(range(window - 1, window - 1 + len(ma)), ma, label=f'Moving Avg (window={window})', color='red',
             linewidth=2)
    plt.xlabel('Simulation Run')
    plt.ylabel('Burned Area (%)')
    plt.title('Monte Carlo Convergence of Burned Area')
    plt.legend()
    plt.grid(True)

    # Plot relative changes
    plt.subplot(2, 1, 2)
    rel_changes = np.abs(np.diff(ma) / (ma[:-1] + 1e-8))
    plt.plot(range(window, window + len(rel_changes)), rel_changes, label='Relative Changes', color='green')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Convergence Threshold (5%)')
    plt.xlabel('Simulation Run')
    plt.ylabel('Relative Change')
    plt.title('Relative Changes in Moving Average')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_single_simulation(hypothesis: str)->tuple[int, int, float]:
    """
    This function runs a single wildfire simulation.
    :param hypothesis: optional string value that determines if the simulation is testing a hypothesis or not
    :return: final burned cells,iterations and percentage.
    """

    sim = WildfireSimulation()
    _, burned = sim.simulate_spread()
    burned_cells = burned[-1]
    total_cells = sim.grid_size * sim.grid_size
    burned_pct = 100 * burned_cells / total_cells
    return (burned_cells, len(burned) - 1, burned_pct)


def run_simulation(n_runs: int=2000, hypothesis: str=None, mc_window: int=20, mc_last_n: int=10, mc_threshold: float=0.05)-> list[tuple[int, int, float]]:

    """
    This function runs multiple wildfire simulations.
    :param n_runs: number of simulation runs
    :param hypothesis: optional string value that determines if the simulation is testing a hypothesis or not
    :param mc_window: window size for convergence check
    :param mc_last_n: last steps for the convergence check
    :param mc_threshold: convergence threshold value
    :return: simulation results (area, iterations, burned percentage)
    """


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

    if hypothesis is None or hypothesis == False:
        print("\nDetailed Summary Statistics:")
        print(f"Burned Area: {mean_area:.2f} ± {std_area:.2f} cells")
        print(f"Iterations: {mean_iter:.2f} ± {std_iter:.2f} steps")
        print(f"Burned Area %: {mean_pct:.2f} ± {std_pct:.2f}%")
        print(f"95% Confidence Interval: [{mean_pct - 1.96 * std_pct:.2f}%, {mean_pct + 1.96 * std_pct:.2f}%]")

        # Monte Carlo convergence check
        converged, ma, rel_changes = monte_carlo_convergence_check(
            percentages,
            window=mc_window,
            last_n=mc_last_n,
            threshold=mc_threshold
        )

        if converged:
            print(f"\nMonte Carlo convergence achieved:")
            print(f"- Relative change < {mc_threshold * 100:.2f}% for last {mc_last_n} steps")
            print(f"- Final moving average: {ma[-1]:.2f}%")
            print(f"- Final relative change: {rel_changes[-1] * 100:.2f}%")
        else:
            print("\nMonte Carlo convergence NOT achieved.")
            print(f"Last {mc_last_n} relative changes:")
            for i, change in enumerate(rel_changes[-mc_last_n:]):
                print(f"  Step {len(rel_changes) - mc_last_n + i + 1}: {change * 100:.2f}%")
    else:
        ma = None

    # Plot histogram of burned areas with normal distribution fit

    plt.figure(figsize=(12, 6))
    plt.hist(percentages, bins=30, color='orange', edgecolor='k', density=True, alpha=0.7)

    # Add normal distribution fit
    x = np.linspace(min(percentages), max(percentages), 100)
    plt.plot(x, norm.pdf(x, mean_pct, std_pct), 'r-', lw=2, label='Normal Distribution Fit')
    # plot_monte_carlo_convergence(list(percentages), ma, mc_window)
    plt.xlabel('Final Burned Area (%)')
    plt.ylabel('Density')
    plt.title('Burned Area Percentage Distribution with Normal Fit')
    plt.grid(True)
    plt.legend()
    plt.savefig('burned_area_distribution.png')
    plt.close()

    # Plot MC convergence
    if hypothesis is None or hypothesis == False:
        plot_cumulative_convergence(percentages)

    if ma is not None:
        plot_monte_carlo_convergence(percentages, ma, window=mc_window)

    return list(zip(areas, iterations, percentages))


def plot_cumulative_convergence(percentages: list[float], path: str='cumulative_convergence.png')-> None:
    """
    Plot cumulative convergence graph
    :param percentages: burned area percentages from simulations
    :param path: output file path
    """
    plt.figure(figsize=(12, 6))
    n = len(percentages)
    pct = np.array(percentages)
    cum_mean = np.cumsum(pct) / (np.arange(1, n + 1))
    cum_std = np.array([pct[:i].std() for i in range(1, n + 1)])
    cum_se = cum_std / np.sqrt(np.arange(1, n + 1))
    plt.axhline(np.mean(percentages), color='red', linestyle='-', label='Final Mean')
    plt.plot(cum_mean, label='Cumulative Mean')
    plt.fill_between(
        np.arange(n),
        cum_mean - cum_se,
        cum_mean + cum_se,
        alpha=0.2,
        label='±1 SE'
    )
    plt.xlabel('Run Index')
    plt.ylabel('Burned Area (%)')
    plt.title('Cumulative Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def hypothesis_testing(control_results: list[tuple[int, int,float]], hypothesis_results: list[tuple[int, int,float]], hypothesis_name: str="")-> None:

    """
    Perform statistical hypothesis testing for control and experiment results
    :param control_results: control(baseline) results
    :param hypothesis_results: hypothesis scenario results
    :param hypothesis_name: name of the hypothesis
    """

    control_burned = [res[2] for res in control_results]
    hypothesis_burned = [res[2] for res in hypothesis_results]

    plt.figure(figsize=(10, 6))
    sns.histplot(control_burned, color='blue', label='Baseline', kde=True, stat="density", bins=30)
    sns.histplot(hypothesis_burned, color='red', label=hypothesis_name, kde=True, stat="density", bins=30, alpha=0.6)
    plt.title(f'Comparison of Burned Area %\nBaseline vs {hypothesis_name}')
    plt.xlabel('Final Burned Area %')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    filename = f"control_vs_{hypothesis_name}_burned_area_comparison.png"
    plt.savefig(filename)

    t_stat, p_value = ttest_ind(control_burned, hypothesis_burned, equal_var=False)

    print(f"\nStatistical Comparison: Control vs {hypothesis_name}")
    print(f"T-statistic = {t_stat:.3f}, P-value = {p_value:.5f}")

    if p_value < 0.05:
        print(f"Result: Significant difference detected at 95% confidence level.")
    else:
        print(f"Result: No significant difference detected at 95% confidence level.")


if __name__ == "__main__":
    # Control Experiment
    print("Running Control Simulation")
    control_results = run_simulation(n_runs=1200, hypothesis=None)  # Increased number of runs for better convergence

    # Hypothesis 1: High humidity
    print("\nRunning Hypothesis 1 Simulation")
    humidity_results = run_simulation(n_runs=1200, hypothesis='high_humidity')

    # Hypothesis 2: High Wind
    print("\nRunning Hypothesis 2 Simulation")
    wind_results = run_simulation(n_runs=1200, hypothesis='high_wind')

    # Hypothesis 3: Dense shrub
    print("\nRunning Hypothesis 3 Simulation")
    shrub_results = run_simulation(n_runs=1200, hypothesis='dense_shrub')

    # Validation (comparison between baseline and each hypothesis)
    hypothesis_testing(control_results, humidity_results, hypothesis_name='High Humidity')
    hypothesis_testing(control_results, wind_results, hypothesis_name='High Wind')
    hypothesis_testing(control_results, shrub_results, hypothesis_name='Dense Shrub')

