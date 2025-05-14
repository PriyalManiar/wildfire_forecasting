# Enhancing Wildfire Forecasting with Monte Carlo Simulations

## Overview
This project simulates wildfire spread over a geographic grid(San diego County)using real environmental data and historical fire occurrences data. It aims to model fire dynamics under varying conditions and assess the impact of specific environmental factors through hypothesis testing.

## Objectives
- Develop a realistic wildfire spread model incorporating environmental variables
- Implement Monte Carlo simulations to analyze fire behavior under stochastic conditions
- Test hypotheses related to environmental influences on wildfire spread

## Methodology

### Data Preparation
- **Ignition Points**: Derived from NASA FIRMS dataset to identify high-frequency fire locations (2012-2025)
- **Environmental Grids**: Generated for elevation, wind speed and direction, humidity, temperature, and vegetation types

#### Environmental Grids Visualization
![Quarterly Humidity Grid](quarterly_humidity_grid.png)
![Quarterly Temperature Grid](quarterly_temperature_grid.png)
![Quarterly Wind Speed Grid](quarterly_wind_speed_grid.png)
![Elevation Grid](elevation_grid.png)

#### Data Sources
1. **Elevation Data**: OpenTopography API
2. **Environmental Variables**: NOAA Website (1st Jan 2012 – 1 May 2025) for stations:
   - Imperial Beach
   - San Diego International Airport
   - Ramona Airport
   - Oceanside

### Simulation Model
- **Grid-Based Approach**: Utilizes a 25x25 grid representing the study area
- **Spread Probability**: Calculated based on local environmental conditions and neighboring cell states
- **Time Evolution**: Simulates fire spread over discrete time steps, updating environmental conditions dynamically

### Data Source Websites
1. NOAA - https://www.ncei.noaa.gov/cdo-web/
2. NASA FIRMS - https://firms.modaps.eosdis.nasa.gov
3. Raw Data: https://drive.google.com/drive/folders/138FxLgycWRDPRQCPdAqJ7gmZvtycBxvd?usp=sharing

## Project Pipeline

### 1. Data Preparation
**Fire Data Processing** (`fire_grid_analysis.py`)
- Extracts fire hotspots from NASA FIRMS archives
- Creates base ignition probability maps
- Outputs: `fire_frequent_cells_25x25_southeast_us.npy`

**Climate Data Integration** (`final_environmental_factors.py`)
- Reads `climate.csv` 
- Normalizes temperature, humidity, and wind data by quarter
- Generates `.npy` grids for simulation use

### 2. Environmental Grid Construction
**Grids Created in `final_environmental_factors.py`:**
- Temperature – Normalized with seasonal variance
- Humidity – AR(1) time series modeling
- Wind – Speed and von Mises-based direction
- Elevation – Fetched from OpenTopoData API
- Output: Normalized 25×25 environmental grids for each factor

### 3. Simulation Engine (`wildfire_simulation_simplify.py`)
**Core Steps:**
1. Loads all environmental grids and ignition points
2. Initializes fire spread matrix
3. Computes fire probability using weighted influence of:
   - Ignition base rate
   - Temperature
   - Humidity
   - Wind speed/direction
   - Elevation
   - Vegetation density

**Fire Spread Algorithm:**
- Accelerated using Numba for efficiency
- Probability-based cell ignition per time step
- Monte Carlo runs to test variability and convergence

## Control Experiment
The control experiment establishes baseline wildfire spread behavior under standard environmental conditions:

- Uses NASA FIRMS-derived ignition points
- Generates environmental grids without hypothesis modifications
- Simulates over 25x25 grid with standard conditions
- Runs 1200 independent Monte Carlo simulations
- Convergence threshold: the moving average of burned area percentages stabilizes with relative changes below 5% for the last 10 steps
- Uses average quarterly environmental conditions

## Hypothesis Testing

### 1. High Humidity
**Hypothesis Statement**: Increased atmospheric humidity levels reduce wildfire spread by lowering the probability of ignition and propagation.

**Null Hypothesis (H0)**: No significant difference in final burned area between control and high humidity (≥65%) simulations.

**Alternative Hypothesis (H1)**: Final burned area is significantly lower in high humidity simulations.

### 2. High Wind Speed
**Hypothesis Statement**: Higher wind speeds accelerate wildfire spread by increasing ignition probability in neighboring cells.

**Null Hypothesis (H0)**: No significant difference in final burned area between control and tripled wind speed simulations.

**Alternative Hypothesis (H1)**: Final burned area is significantly higher in high wind speed simulations.

### 3. Dense Shrub Vegetation
**Hypothesis Statement**: Higher shrub density increases wildfire ignition likelihood and total spread due to greater fuel availability.

**Null Hypothesis (H0)**: No significant difference in final burned area between control and increased shrub density (2.5x) simulations.

**Alternative Hypothesis (H1)**: Final burned area is significantly higher in dense shrub vegetation simulations.

## Key Technical Highlights

### Performance
- Numba acceleration for critical computation paths
- Multiprocessing for parallel Monte Carlo simulations
- Vectorized operations for grid calculations

### Environmental Realism
- AR(1) time series for weather evolution
- Von Mises distribution for wind direction
- Seasonal variation in climate parameters

### Modeling
- Grid-based probabilistic simulation
- Weighted environmental factor influence
- Monte Carlo convergence analysis

## Project Execution Flow

### Step 1: Data Preparation and Analysis
```bash
# First, process NASA FIRMS data to identify fire hotspots
python fire_grid_analysis.py
# This generates: fire_frequent_cells_25x25_southeast_us.npy
```

### Step 2: Environmental Grid Generation
```bash
# Generate all environmental grids (temperature, humidity, wind, elevation)
python final_environmental_factors.py
```

### Step 3: Run Simulations
```bash
# Run the main simulation with different hypotheses
python wildfire_simulations_simplify.py
# This will:
# 1. Run control experiment (1200 simulations)
# 2. Test high humidity hypothesis
# 3. Test high wind hypothesis
# 4. Test dense shrub hypothesis
# 5. Generate comparison plots and statistical analysis
```

## Sample Code Snippets

### Fire Spread Probability
```python
prob = (
    weights[0] * base +
    weights[1] * norm_temp +
    weights[2] * (1 - norm_humidity) +
    weights[3] * norm_wind_speed +
    weights[4] * (1 - norm_elevation)
)
```

### Weather AR(1) Update
```python
new_value = persistence * current_value + variability * random_draw
```

### Wind Direction Weighting
```python
wind_influence = max(0.0, np.cos(np.radians(angle_diff)))
```

## Limitations
1. Grid Resolution: The 25x25 grid may not capture finer-scale fire dynamics
2. Simplified Assumptions: No long-range ember transport or complex terrain effects
3. Weather modeled via AR(1), not full spatiotemporal simulation
4. Fixed vegetation assumptions per grid cell
5. Hypothesis Impact: Tested hypotheses did not yield statistically significant differences

## Dependencies
- NumPy
- Pandas
- Matplotlib
- SciPy
- Numba
