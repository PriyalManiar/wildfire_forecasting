# Enhancing Wildfire Forecasting with Monte Carlo Simulations


## Overview

This project simulates wildfire spread over a geographic grid using environmental data and historical fire occurrences data. It aims to model fire dynamics under varying conditions and assess the impact of specific environmental factors through hypothesis testing.

## Objectives

•	Develop a realistic wildfire spread model incorporating environmental variables.

•	Implement Monte Carlo simulations to analyze fire behavior under stochastic conditions.

•	Test hypotheses related to environmental influences on wildfire spread.

## Methodology

### Data Preparation

•	Ignition Points: Derived from NASA FIRMS dataset to identify high-frequency fire locations. ( 2012-2025)


•	Environmental Grids: Generated for elevation, wind speed and direction, humidity, temperature, and vegetation types.

<img width="375" alt="quarterly_humidity_grid" src="https://github.com/user-attachments/assets/a9c9bde7-9234-4372-87be-47c7aa8653f7" />

<img width="375" alt="quarterly_temperature_grid" src="https://github.com/user-attachments/assets/76fd084b-ae23-4b82-83f3-9074ba57e1ab" />

<img width="375" alt="quarterly_wind_speed_grid" src="https://github.com/user-attachments/assets/39b3aa4c-3140-41de-94eb-4f79c3b65fb6" />


<img width="375" alt="elevation_grid" src="https://github.com/user-attachments/assets/e5dc862d-9d6b-4da5-8f6e-9407104a3781" />


a. Elevation Data- OpenTopography API

b. Other Environmental Variables- NOAA Website (1st Jan 2012 – 1 May 2025) for stations : Imperial Beach, San Diego International Airport, Ramona Airport and Oceanside.
Simulation Model


•	Grid-Based Approach: Utilizes a 25x25 grid representing the study area.

•	Spread Probability: Calculated based on local environmental conditions and neighboring cell states.

•	Time Evolution: Simulates fire spread over discrete time steps, updating environmental conditions dynamically.

• Data Source Website: (all data was requested using the order data feature on these websites)

    1. NOAA - https://www.ncei.noaa.gov/cdo-web/
    
    2. NASA FIRMS - https://firms.modaps.eosdis.nasa.gov

## Monte Carlo Simulations

•	Parallel Processing: Employs multiprocessing to run multiple simulations concurrently.

•	Convergence Analysis: Uses moving averages and relative change metrics to assess simulation stability.

## Control Experiment

The control experiment establishes the baseline behavior of the wildfire spread simulation under standard environmental conditions. It uses ignition points derived from the NASA FIRMS dataset and generates environmental grids for elevation, wind speed and direction, temperature, humidity, and vegetation ignition probabilities without any modifications related to the tested hypotheses.

The wildfire spread is simulated over a 25 by 25 grid. Spread probabilities are calculated at each time step based on conditions including vegetation type, wind influence, elevation gradient, temperature, and humidity. The simulation proceeds iteratively until convergence is detected based on a relative change threshold in burned area over successive iterations. Convergence is considered achieved when the relative change in burned area remains below 1 percent over the last five time steps.

For the control case, average values are chosen for the environmental conditions. Wind speed, humidity, temperature, and vegetation parameters are sampled from historical distributions and represent average quarterly conditions. 

The control experiment is run for independent Monte Carlo simulations, 1200 runs to ensure statistical stability. The control results provide a reference for the final burned areas, which is later used to evaluate the effects of environmental changes introduced under the hypothesis testing scenarios.



## Hypothesis Testing

Three hypotheses were tested:

1. ### High Humidity

Hypothesis Statement: Increased atmospheric humidity levels reduce wildfire spread by lowering the probability of ignition and propagation.

Null Hypothesis (H0): There is no significant difference in the final burned area between the control simulations and the simulations with increased minimum humidity levels (65 percent or higher) across the grid.

Alternative Hypothesis (H1): The final burned area is significantly lower in simulations with increased minimum humidity levels (65 percent or higher) compared to the control simulations.

2. ### High Wind Speed

Hypothesis Statement: Higher wind speeds accelerate wildfire spread by increasing the probability of ignition in neighboring cells.

Null Hypothesis (H0): There is no significant difference in the final burned area between the control simulations and the simulations with wind speed values tripled.

Alternative Hypothesis (H1): The final burned area is significantly higher in simulations with tripled wind speed values compared to the control simulations.

3. ### Dense Shrub Vegetation

Hypothesis Statement: Higher shrub density increases wildfire ignition likelihood and total spread due to greater fuel availability.

Null Hypothesis (H0): There is no significant difference in the final burned area between the control simulations and the simulations with increased ignition probabilities for dense shrub vegetation (boosted by a factor of 2.5).

Alternative Hypothesis (H1): The final burned area is significantly higher in simulations with increased ignition probabilities for dense shrub vegetation compared to the control simulations.

Statistical analyses, including t-tests, were conducted to compare the outcomes of these scenarios against the control simulations.

## Key Features

•	Numba Integration: Accelerates computation of spread probabilities through just-in-time compilation.

•	Parallel Processing: Enhances simulation throughput using Python's multiprocessing capabilities.

•	Statistical Analysis: Provides quantitative assessment of hypothesis impacts on fire spread.

•	Visualization Tools: Generates plots for burned area over time, convergence behavior, and distribution comparisons.

## Limitations

•	Grid Resolution: The 25x25 grid may not capture finer-scale fire dynamics.

•	Simplified Assumptions: The model does not account for long-range ember transport or complex terrain effects.

•	Hypothesis Impact: The tested hypotheses did not yield statistically significant differences, possibly due to model constraints.
