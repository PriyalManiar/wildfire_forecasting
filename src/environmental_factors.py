import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
import os
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from scipy.stats import truncnorm, norm

class EnvironmentalFactors:
    def __init__(self, grid_size=25, min_lat=32.5, max_lat=33.5, min_lon=-117.6, max_lon=-116.1, dataset="etopo1"):
        self.grid_size = grid_size
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.dataset = dataset
        self.lat_grid = np.linspace(min_lat, max_lat, grid_size + 1)
        self.lon_grid = np.linspace(min_lon, max_lon, grid_size + 1)
        self.quarter = self._select_random_quarter()

    def _select_random_quarter(self):
        """Randomly select a quarter (Q1-Q4) for the simulation."""
        return f'Q{np.random.randint(1, 5)}'
    
    def _get_quarter_data(self, df, column, quarter):
        """Get data for the selected quarter from the monthly data."""
        quarter_data = df[df['Quarter'] == quarter][column]
        quarter_data = quarter_data.astype(str)
        valid_data = quarter_data[~quarter_data.str.endswith('s')]
        valid_data = pd.to_numeric(valid_data, errors='coerce').dropna().values
        if len(valid_data) == 0:
            raise ValueError(f"No valid data found for quarter {quarter} in column {column}")
        return valid_data

    def generate_elevation_grid(self):
        lat_grid = np.linspace(self.min_lat, self.max_lat, self.grid_size)
        lon_grid = np.linspace(self.min_lon, self.max_lon, self.grid_size)
        locations = [f"{lat:.5f},{lon:.5f}" for lat in lat_grid for lon in lon_grid]

        elevation_grid = []
        for i in range(0, len(locations), 100):
            batch = "|".join(locations[i:i + 100])
            url = f"https://api.opentopodata.org/v1/{self.dataset}?locations={batch}"
            response = requests.get(url)
            if response.status_code == 200:
                results = response.json()["results"]
                elevation_grid.extend([r["elevation"] for r in results])
            else:
                print("API error:", response.status_code)
            time.sleep(1)

        elevation_array = np.array(elevation_grid).reshape(len(lat_grid), len(lon_grid))
        plt.imshow(elevation_array, cmap='terrain', origin='lower')
        plt.colorbar(label='Elevation (m)')
        plt.title("Elevation Grid")
        plt.xlabel("Longitude Index")
        plt.ylabel("Latitude Index")
        return elevation_array

    def generate_wind_grids(self, quarter, csv_path='climate.csv'):
        df = pd.read_csv(csv_path)
        wind_speed_grid = np.zeros((self.grid_size, self.grid_size))
        wind_dir_grid = np.zeros((self.grid_size, self.grid_size))
        wind_speed_values = self._get_quarter_data(df, 'DailyAverageWindSpeed', quarter)
        wind_dir_values = self._get_quarter_data(df, 'DailyPeakWindDirection', quarter)
        # Fit normal distributions
        mu_speed, sigma_speed = norm.fit(wind_speed_values)
        mu_dir, sigma_dir = norm.fit(wind_dir_values)
        # Sample from fitted distributions
        wind_speed_grid = np.clip(norm.rvs(mu_speed, sigma_speed, size=(self.grid_size, self.grid_size)), 0, None)
        wind_dir_grid = np.clip(norm.rvs(mu_dir, sigma_dir, size=(self.grid_size, self.grid_size)), 0, 360)
        np.save('wind_speed_grid_25x25.npy', wind_speed_grid)
        np.save('wind_dir_grid_25x25.npy', wind_dir_grid)
        print(f'Saved: wind_speed_grid_25x25.npy and wind_dir_grid_25x25.npy (Quarter {quarter})')
        return wind_speed_grid, wind_dir_grid

    # def generate_humidity_grid(self, quarter, csv_path='climate.csv'):
    #     df = pd.read_csv(csv_path)
    #     humidity_values = self._get_quarter_data(df, 'DailyAverageRelativeHumidity', quarter)
    #     mu, sigma = norm.fit(humidity_values)
    #     humidity_grid = np.clip(norm.rvs(mu, sigma, size=(self.grid_size, self.grid_size)), 0, 100)
    #     np.save('humidity_grid_25x25.npy', humidity_grid)
    #     plt.figure(figsize=(7, 6))
    #     plt.imshow(humidity_grid, cmap='Blues', origin='lower',
    #               extent=[self.min_lon, self.max_lon, self.min_lat, self.max_lat],
    #               aspect='auto')
    #     plt.colorbar(label='Relative Humidity (%)')
    #     plt.title(f'Quarterly Humidity Grid (Quarter {quarter})')
    #     plt.xlabel('Longitude')
    #     plt.ylabel('Latitude')
    #     plt.tight_layout()
    #     plt.savefig('humidity_grid_25x25.png')
    #     plt.close()
    #     print(f'Saved: humidity_grid_25x25.png (Quarter {quarter})')
    #     return humidity_grid

    def generate_humidity_grid(self, quarter, high_humidity=False, csv_path='climate.csv'):
        df = pd.read_csv(csv_path)
        humidity_values = self._get_quarter_data(df, 'DailyAverageRelativeHumidity', quarter)
        mu, sigma = norm.fit(humidity_values)

        if high_humidity:
            # Truncate normal distribution to values > 75%
            lower, upper = 75, 100
            a, b = (lower - mu) / sigma, (upper - mu) / sigma
            humidity_grid = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=(self.grid_size, self.grid_size))
        else:
            # Sample from full normal distribution, then clip to 0–100% range
            humidity_grid = norm.rvs(mu, sigma, size=(self.grid_size, self.grid_size))
            humidity_grid = np.clip(humidity_grid, 0, 100)

        suffix = 'high' if high_humidity else 'baseline'
        np.save(f'humidity_grid_25x25_{suffix}.npy', humidity_grid)

        plt.figure(figsize=(7, 6))
        plt.imshow(humidity_grid, cmap='Blues', origin='lower',
                   extent=[self.min_lon, self.max_lon, self.min_lat, self.max_lat],
                   aspect='auto')
        plt.colorbar(label='Relative Humidity (%)')
        plt.title(f'Quarterly Humidity Grid (Quarter {quarter}) - {suffix.capitalize()}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        plt.savefig(f'humidity_grid_25x25_{suffix}.png')
        plt.close()

        print(f'Saved: humidity_grid_25x25_{suffix}.png (Quarter {quarter})')
        return humidity_grid

    def generate_vegetation_ignition_grid(self, folder='vegetation', hypothesis_dense_shrub=False,density_factor=1):
        ignition_probs = {
            "Openshrub": 0.4, "C3_grass": 0.6, "C3past": 0.3,
            "DenseShrub": 0.5, "SecTmpENF": 0.3, "tmpENF": 0.6,
            "urban": 0.2, "water": 0.0
        }
        ignition_grid = np.zeros((self.grid_size, self.grid_size))

        for veg_key, ign_prob in ignition_probs.items():
            file_path = os.path.join(folder, f"{veg_key}.csv")
            df = pd.read_csv(file_path)
            veg_grid = np.zeros((self.grid_size, self.grid_size))

            for _, row in df.iterrows():
                try:
                    lat, lon, perc = float(row['latitude']), float(row['longitude']), float(row['percentage'])
                    if lon > 180:
                        lon -= 360
                    if pd.isna(perc):
                        continue
                    if veg_key == "DenseShrub" and hypothesis_dense_shrub:
                        perc = min(perc * 1.10, 100)  # Increase by 10%, cap at 100%
                    i = np.digitize(lat, self.lat_grid) - 1
                    j = np.digitize(lon, self.lon_grid) - 1
                    if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                        veg_grid[i, j] = perc
                except:
                    continue


            ignition_grid += (veg_grid / 100.0) * ign_prob

        ignition_grid *= 1  # Assuming density_factor is 1

        # Ensure the ignition grid values stay within the [0, 1] range
        ignition_grid = np.clip(ignition_grid, 0, 1)

        np.save('ignition_probability_25x25.npy', ignition_grid)
        plt.imshow(ignition_grid, cmap='OrRd', origin='lower',
                   extent=[self.min_lon, self.max_lon, self.min_lat, self.max_lat], aspect='auto', vmin=0, vmax=1)
        plt.colorbar(label='Ignition Probability')
        plt.title('Vegetation-Based Ignition Probability')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig('ignition_probability_25x25.png')
        plt.close()
        print('Saved: ignition_probability_25x25.png')
        return ignition_grid

    def temperature_grid(self, quarter):
        df = pd.read_csv('climate.csv')
        if 'DailyAverageDryBulbTemperature' not in df.columns:
            raise ValueError("'DailyAverageDryBulbTemperature' column not found in climate.csv")
        temp_values = self._get_quarter_data(df, 'DailyAverageDryBulbTemperature', quarter)
        mu, sigma = norm.fit(temp_values)
        grid = norm.rvs(mu, sigma, size=(self.grid_size, self.grid_size))
        np.save('temperature_grid_25x25.npy', grid)
        plt.figure(figsize=(7, 6))
        plt.imshow(grid, cmap='Reds', origin='lower',
                  extent=[self.min_lon, self.max_lon, self.min_lat, self.max_lat],
                  aspect='auto')
        plt.colorbar(label='Temperature (°C)')
        plt.title(f'Quarterly Temperature Grid (Quarter {quarter})')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        plt.savefig('temperature_grid_25x25.png')
        plt.close()
        print(f'Saved: temperature_grid_25x25.png (Quarter {quarter})')
        return grid

def main(hypothesis_dense_shrub=False):
    env = EnvironmentalFactors(grid_size=25, min_lat=32.5, max_lat=33.5, min_lon=-117.6, max_lon=-116.1)
    print(f"Selected Quarter: {env.quarter}")

    print("Generating Elevation Grid...")
    elevation = env.generate_elevation_grid()

    print("Generating Wind Grids...")
    wind_speed, wind_dir = env.generate_wind_grids(env.quarter)

    print("Generating Humidity Grid...")
    humidity = env.generate_humidity_grid(env.quarter)

    print("Generating Vegetation Ignition Probability Grid...")
    ignition = env.generate_vegetation_ignition_grid(hypothesis_dense_shrub=hypothesis_dense_shrub)

    print("Generating Temperature Grid...")
    temperature = env.temperature_grid(env.quarter)

if __name__ == "__main__":
    main()
