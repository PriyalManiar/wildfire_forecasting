**Enhancing Wildfire Forecasting with Monte Carlo Simulations**


**Overview**

This project simulates wildfire spread over a geographic grid using environmental data and historical fire occurrences data. It aims to model fire dynamics under varying conditions and assess the impact of specific environmental factors through hypothesis testing.

**Objectives**

•	Develop a realistic wildfire spread model incorporating environmental variables.

•	Implement Monte Carlo simulations to analyze fire behavior under stochastic conditions.

•	Test hypotheses related to environmental influences on wildfire spread.

**Methodology**

_Data Preparation_

•	Ignition Points: Derived from NASA FIRMS dataset to identify high-frequency fire locations. ( 2012-2025)

•	Environmental Grids: Generated for elevation, wind speed and direction, humidity, temperature, and vegetation types.

o	Elevation Data: OpenTopography API

o	Other Environmental Variables: NOAA Website (1st Jan 2012 – 1 May 2025) for stations : Imperial Beach, San Diego International Airport, Ramona Airport and Oceanside.
Simulation Model

•	Grid-Based Approach: Utilizes a 25x25 grid representing the study area.

•	Spread Probability: Calculated based on local environmental conditions and neighboring cell states.

•	Time Evolution: Simulates fire spread over discrete time steps, updating environmental conditions dynamically.

**Monte Carlo Simulations**

•	Parallel Processing: Employs multiprocessing to run multiple simulations concurrently.

•	Convergence Analysis: Uses moving averages and relative change metrics to assess simulation stability.

**Hypothesis Testing**

Three hypotheses were tested:

1.	High Humidity: Increased minimum humidity levels across the grid.
	
2.	High Wind Speed: Tripled wind speed values to assess impact on fire spread.
	
3.	Dense Shrub Vegetation: Increased ignition probability for dense shrub areas.
	

Statistical analyses, including t-tests, were conducted to compare the outcomes of these scenarios against the control simulations.

**Key Features**

•	Numba Integration: Accelerates computation of spread probabilities through just-in-time compilation.

•	Parallel Execution: Enhances simulation throughput using Python's multiprocessing capabilities.

•	Statistical Analysis: Provides quantitative assessment of hypothesis impacts on fire spread.

•	Visualization Tools: Generates plots for burned area over time, convergence behavior, and distribution comparisons.

**Limitations**

•	Grid Resolution: The 25x25 grid may not capture finer-scale fire dynamics.

•	Simplified Assumptions: The model does not account for long-range ember transport or complex terrain effects.

•	Hypothesis Impact: The tested hypotheses did not yield statistically significant differences, possibly due to model constraints.
