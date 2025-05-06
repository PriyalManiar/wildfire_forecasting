import numpy as np
import pandas as pd

# Grid boundaries and resolution (match fire_grid_analysis)
GRID_SIZE = 25

# Read wind data
wind_df = pd.read_csv('monthly_averages.csv')
wind_df = wind_df.dropna(subset=['MonthlyAverageWindSpeed', 'MonthlyAverageWindDirection'])

# Prepare arrays for wind speed and direction
wind_speed_grid = np.zeros((GRID_SIZE, GRID_SIZE))
wind_dir_grid = np.zeros((GRID_SIZE, GRID_SIZE))

# For each cell, randomly pick a row and use both speed and direction from that row
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        row = wind_df.sample(1).iloc[0]
        wind_speed_grid[i, j] = row['MonthlyAverageWindSpeed']
        wind_dir_grid[i, j] = row['MonthlyAverageWindDirection']

# Save grids for use in modeling
np.save('wind_speed_grid_25x25.npy', wind_speed_grid)
np.save('wind_dir_grid_25x25.npy', wind_dir_grid)

print('Saved: wind_speed_grid_25x25.npy and wind_dir_grid_25x25.npy')