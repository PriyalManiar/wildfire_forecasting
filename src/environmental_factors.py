import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
import os
from scipy.ndimage import gaussian_filter

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
        plt.show()
        return elevation_array

    def generate_wind_grids(self, csv_path='monthly_averages.csv'):
        df = pd.read_csv(csv_path).dropna(subset=['MonthlyAverageWindSpeed', 'MonthlyAverageWindDirection'])
        wind_speed_grid = np.zeros((self.grid_size, self.grid_size))
        wind_dir_grid = np.zeros((self.grid_size, self.grid_size))

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                row = df.sample(1).iloc[0]
                wind_speed_grid[i, j] = row['MonthlyAverageWindSpeed']
                wind_dir_grid[i, j] = row['MonthlyAverageWindDirection']

        np.save('wind_speed_grid_25x25.npy', wind_speed_grid)
        np.save('wind_dir_grid_25x25.npy', wind_dir_grid)
        print('Saved: wind_speed_grid_25x25.npy and wind_dir_grid_25x25.npy')
        return wind_speed_grid, wind_dir_grid

    def generate_humidity_grid(self, csv_path='monthly_averages.csv'):
        
        df = pd.read_csv(csv_path)
        humidity_values = df['MonthlyAverageRelativeHumidity'].dropna().values

        humidity_grid = np.random.choice(humidity_values, size=(self.grid_size, self.grid_size))

        np.save('humidity_grid_25x25.npy', humidity_grid)

        plt.figure(figsize=(7, 6))
        plt.imshow(humidity_grid, cmap='Blues', origin='lower', extent=[self.min_lon, self.max_lon, self.min_lat, self.max_lat], aspect='auto')
        plt.colorbar(label='Relative Humidity (%)')
        plt.title('Randomly Sampled Humidity Grid (25x25)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        plt.savefig('humidity_grid_25x25.png')
        plt.close()
        print('Saved: humidity_grid_25x25.png')
        
        return humidity_grid

    def generate_vegetation_ignition_grid(self, folder='vegetation'):
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
                    i = np.digitize(lat, self.lat_grid) - 1
                    j = np.digitize(lon, self.lon_grid) - 1
                    if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                        veg_grid[i, j] = perc
                except:
                    continue

            ignition_grid += (veg_grid / 100.0) * ign_prob

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

    def temperature_grid(self):
        """Generate a 25x25 grid of temperature values randomly sampled from the CSV."""
        df = pd.read_csv('monthly_averages.csv')
        if 'MonthlyAverageTemperature' not in df.columns:
            raise ValueError("'MonthlyAverageTemperature' column not found in monthly_averages.csv")
        temp_values = df['MonthlyAverageTemperature'].dropna().values
        grid = np.random.choice(temp_values, size=(self.grid_size, self.grid_size))
        return grid
    

def main():
    env = EnvironmentalFactors(grid_size=25, min_lat=32.5, max_lat=33.5, min_lon=-117.6, max_lon=-116.1)

    print("Generating Elevation Grid...")
    elevation = env.generate_elevation_grid()

    print("Generating Wind Grids...")
    wind_speed, wind_dir = env.generate_wind_grids()

    print("Generating Humidity Grid...")
    humidity = env.generate_humidity_grid()

    print("Generating Vegetation Ignition Probability Grid...")
    ignition = env.generate_vegetation_ignition_grid()

    print("Generating Temperature Grid...")
    temperature = env.temperature_grid()
    np.save('temperature_grid_25x25.npy', temperature)
    print("Saved: temperature_grid_25x25.npy")

if __name__ == "__main__":
    main()

