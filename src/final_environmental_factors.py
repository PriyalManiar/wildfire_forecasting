import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
import os
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

class EnvironmentalFactors:
    """
    Defines methods to generate environmental grids - elevation, wind, humidity,
    vegetation ignition probability, and temperature) for a given geographic area and quarter.
    """
    # Shared class variables to cache grids if needed
    _elevation_grid = None
    _vegetation_grid = None

    def __init__(
        self,
        grid_size=25,
        min_lat=32.5,
        max_lat=33.5,
        min_lon=-117.6,
        max_lon=-116.1,
        dataset="etopo1"
    )-> None:
        """
        Initialize geographic bounds and grid resolution.

        :param grid_size: Number of grid points along each dimension.
        :param min_lat: Minimum latitude.
        :param max_lat: Maximum latitude.
        :param min_lon: Minimum longitude.
        :param max_lon: Maximum longitude.
        :param dataset: Dataset identifier for elevation data source.

        """
        self.grid_size = grid_size
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.dataset = dataset
        # Create latitude and longitude edges for digitization
        self.lat_grid = np.linspace(min_lat, max_lat, grid_size + 1)
        self.lon_grid = np.linspace(min_lon, max_lon, grid_size + 1)
        # Randomly pick a quarterly season for climate data
        self.quarter = self._select_random_quarter()

    def _select_random_quarter(self)-> str:
        """
        Randomly selects one of the 4 quarters : Q1, Q2, Q3, Q4

        :return: A string representing the randomly selected quarter.
        """
        return f'Q{np.random.randint(1, 5)}'

    def _get_quarter_data(self, df: pd.DataFrame, column:str, quarter:str)->np.ndarray:
        """
        Extract numeric values for a given quarter from a DataFrame.

        :param df: Pandas DataFrame containing the data.
        :param column: Name of the column to extract data from.
        :param quarter: Name of the quarter to extract data from.
        :return: Numpy array of numeric quarter values

        """
        # Select rows matching the quarter
        quarter_data = df[df['Quarter'] == quarter][column].astype(str)
        # Remove entries ending with letters (e.g., '30s')
        valid = quarter_data[~quarter_data.str.endswith('s')]
        # Convert to numeric, drop NaNs
        values = pd.to_numeric(valid, errors='coerce').dropna().values
        if len(values) == 0:
            raise ValueError(f"No valid data for quarter {quarter} in column {column}")
        return values

    def generate_elevation_grid(self)-> np.ndarray:
        """
        Generate and load an elevation grid for the selected region using an API call

        ":return: 2D Numpy array of elevation values
        """
        npy_file = 'elevation_grid_25x25.npy'
        # If we saved it before, just load
        if os.path.exists(npy_file):
            return np.load(npy_file)

        print("Generating elevation grid; this may take a while...")
        # Prepare locations as lat,lon strings
        lats = np.linspace(self.min_lat, self.max_lat, self.grid_size)
        lons = np.linspace(self.min_lon, self.max_lon, self.grid_size)
        locations = [f"{lat:.5f},{lon:.5f}" for lat in lats for lon in lons]

        elevation_list = []
        for i in range(0, len(locations), 100):
            # Batch requests to avoid URL length limits
            batch = "|".join(locations[i:i+100])
            url = f"https://api.opentopodata.org/v1/{self.dataset}?locations={batch}"
            # Retry logic for rate limiting or failures
            for attempt in range(3):
                resp = requests.get(url)
                if resp.status_code == 200:
                    data = resp.json().get('results', [])
                    elevation_list.extend([r['elevation'] for r in data])
                    break
                elif resp.status_code == 429:
                    time.sleep(2 * (attempt + 1))  # exponential backoff
                else:
                    time.sleep(1)
            time.sleep(1)  # gentle rate limit

        # Ensure we got full grid
        expected = self.grid_size ** 2
        if len(elevation_list) != expected:
            raise ValueError(f"Got {len(elevation_list)} points, expected {expected}")

        arr = np.array(elevation_list).reshape(self.grid_size, self.grid_size)
        np.save(npy_file, arr)
        # Optional: save a plot for visual check
        plt.imshow(arr, cmap='terrain', origin='lower')
        plt.colorbar(label='Elevation (m)')
        plt.title('Elevation Grid')
        plt.savefig('elevation_grid.png')
        plt.close()
        return arr

    def generate_wind_grids(self, quarter:str, csv_path:str='climate.csv')-> tuple[np.ndarray, np.ndarray]:
        """
        Create wind speed and direction grids for the selected quarter.

        :param quarter: Name of the quarter to extract data from.
        :param csv_path: Path to the csv file.
        :return: Tuple 2 numpy arrays : wind speed grid (in m/s) and wind direction grid (in degrees)

        """
        df = pd.read_csv(csv_path)
        # Extract raw values for stats
        speeds = self._get_quarter_data(df, 'DailyAverageWindSpeed', quarter)
        dirs = self._get_quarter_data(df, 'DailyPeakWindDirection', quarter)
        # Fit and sample
        mu_s, sd_s = norm.fit(speeds)
        mu_d, sd_d = norm.fit(dirs)
        speed_grid = np.clip(norm.rvs(mu_s, sd_s, size=(self.grid_size, self.grid_size)), 0, None)
        dir_grid = np.mod(norm.rvs(mu_d, sd_d, size=(self.grid_size, self.grid_size)), 360)
        # Cache for reuse
        np.save('wind_speed_grid_25x25.npy', speed_grid)
        np.save('wind_dir_grid_25x25.npy', dir_grid)

        plt.figure(figsize=(6, 5))
        plt.imshow(speed_grid, cmap='Greens', origin='lower',
                   extent=[self.min_lon, self.max_lon, self.min_lat, self.max_lat])
        plt.colorbar(label='Wind Speed')
        plt.title(f'Wind Speed Grid (Quarter {quarter})')
        plt.savefig('wind_speed_grid.png')
        plt.close()

        return speed_grid, dir_grid

    def generate_humidity_grid(self, quarter:str, csv_path:str='climate.csv')-> np.ndarray:
        """
        Generate a grid of humidity levels for the selected quarter.

        :param quarter: Name of the quarter to extract data from.
        :param csv_path: Path to the csv file for climate data
        :return: 2D numpy arrays with humidity values
        """
        df = pd.read_csv(csv_path)
        hums = self._get_quarter_data(df, 'DailyAverageRelativeHumidity', quarter)
        mu, sd = norm.fit(hums)
        grid = np.clip(norm.rvs(mu, sd, size=(self.grid_size, self.grid_size)), 0, 100)
        np.save('humidity_grid_25x25.npy', grid)

        plt.figure(figsize=(6,5))
        plt.imshow(grid, cmap='Blues', origin='lower',
                   extent=[self.min_lon, self.max_lon, self.min_lat, self.max_lat])
        plt.colorbar(label='Humidity (%)')
        plt.title(f'Humidity Grid (Quarter {quarter})')
        plt.savefig('humidity_grid.png')
        plt.close()
        return grid

    def generate_vegetation_ignition_grid(
        self,
        folder: str='vegetation',
        hypothesis_dense_shrub: bool=False,
        density_factor: float=1
    )-> np.ndarray:
        """
        Generate a vegetation ignition grid for the selected quarter based on vegetation type data.

        :param folder: Path to the folder where the data will be saved.
        :param hypothesis_dense_shrub: if true, increases DenseShrub ignition probability
        :param density_factor: Scaling factor for overall ignition probabilities
        :return: 2D numpy arrays with vegetation ignition grid

        """
        npy_file = 'vegetation_ignition_grid_25x25.npy'
        if os.path.exists(npy_file):
            return np.load(npy_file)

        # Base ignition likelihood per vegetation class
        base_probs = {
            'Openshrub':0.4, 'C3_grass':0.6, 'C3past':0.3,
            'DenseShrub':0.5, 'SecTmpENF':0.3, 'tmpENF':0.6,
            'urban':0.2, 'water':0.0
        }
        grid = np.zeros((self.grid_size, self.grid_size))

        # Loop through types
        for veg, prob in base_probs.items():
            path = os.path.join(folder, f"{veg}.csv")
            df = pd.read_csv(path)
            subgrid = np.zeros_like(grid)
            # Map each record to a cell
            for _, row in df.iterrows():
                try:
                    lat, lon, pct = row['latitude'], row['longitude'], row['percentage']
                    if lon > 180: lon -= 360
                    # Optionally boost DenseShrub
                    if veg=='DenseShrub' and hypothesis_dense_shrub:
                        pct = min(pct*2.5, 100)
                    i = np.digitize(lat, self.lat_grid)-1
                    j = np.digitize(lon, self.lon_grid)-1
                    if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                        subgrid[i,j] = pct/100.0
                except:
                    continue
            grid += subgrid * prob

        # Apply overall density factor and clip
        grid = np.clip(grid * density_factor, 0, 1)
        np.save(npy_file, grid)
        plt.imshow(grid, cmap='OrRd', origin='lower',
                   extent=[self.min_lon, self.max_lon, self.min_lat, self.max_lat])
        plt.colorbar(label='Ignition Prob.')
        plt.title('Vegetation Ignition Probability')
        plt.savefig('vegetation_ignition.png')
        plt.close()
        return grid

    def temperature_grid(self, quarter:str, csv_path:str='climate.csv')-> np.ndarray:
        """
        Generate a temperature grid for the selected quarter based on temperature data.

        :param quarter: Quarter to generate grid for selected quarter.
        :param csv_path: Path to the csv file containing climate data.
        :return: 2D numpy arrays with temperature values
        """
        df = pd.read_csv(csv_path)
        vals = self._get_quarter_data(df, 'DailyAverageDryBulbTemperature', quarter)
        mu, sd = norm.fit(vals)
        grid = norm.rvs(mu, sd, size=(self.grid_size, self.grid_size))
        np.save('temperature_grid_25x25.npy', grid)
        plt.imshow(grid, cmap='Reds', origin='lower',
                   extent=[self.min_lon, self.max_lon, self.min_lat, self.max_lat])
        plt.colorbar(label='Temperature (°C)')
        plt.title(f'Temperature Grid (Quarter {quarter})')
        plt.savefig('temperature_grid.png')
        plt.close()
        return grid

# Example main to generate all factors at once
if __name__ == '__main__':
    env = EnvironmentalFactors()
    print(f"Using quarter: {env.quarter}")
    elev = env.generate_elevation_grid()
    ws, wd = env.generate_wind_grids(env.quarter)
    hum = env.generate_humidity_grid(env.quarter)
    veg = env.generate_vegetation_ignition_grid()
    temp = env.temperature_grid(env.quarter)
    print("All environmental grids generated.")
