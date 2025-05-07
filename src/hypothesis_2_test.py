import numpy as np
import matplotlib.pyplot as plt
from wildfire_simulation import WildfireSimulation, plot_convergence

def run_hypothesis_2_test(n_runs=100):
    """
    Run hypothesis 2 testing with proper handling of environmental conditions.
    For each run, we'll run both control and hypothesis simulations with the same
    environmental conditions in each iteration.
    """
    control_results = []
    hypothesis_results = []
    
    print("Running Hypothesis 2 Testing...")
    for run in range(n_runs):
        if run % 10 == 0:
            print(f"Run {run + 1}/{n_runs}")
            
        # Run control simulation
        control_sim = WildfireSimulation(hypothesis_dense_shrub=False)
        control_grid, control_burned = control_sim.simulate_spread()
        control_results.append({
            'final_burned': control_burned[-1],
            'iterations': len(control_burned) - 1,
            'burned_over_time': control_burned
        })
        
        # Run hypothesis simulation with same environmental conditions
        hypo_sim = WildfireSimulation(hypothesis_dense_shrub=True)
        # Copy environmental conditions from control
        hypo_sim.elevation = control_sim.elevation
        hypo_sim.environmental_history = control_sim.environmental_history.copy()
        
        hypo_grid, hypo_burned = hypo_sim.simulate_spread()
        hypothesis_results.append({
            'final_burned': hypo_burned[-1],
            'iterations': len(hypo_burned) - 1,
            'burned_over_time': hypo_burned
        })
    
    return control_results, hypothesis_results

def analyze_results(control_results, hypothesis_results):
    """Analyze and visualize the results of hypothesis 2 testing."""
    
    # Calculate statistics
    control_burned = [r['final_burned'] for r in control_results]
    hypothesis_burned = [r['final_burned'] for r in hypothesis_results]
    
    print("\n=== Hypothesis 2 Test Results ===")
    print(f"Control - Average burned area: {np.mean(control_burned):.2f} ± {np.std(control_burned):.2f}")
    print(f"Hypothesis 2 - Average burned area: {np.mean(hypothesis_burned):.2f} ± {np.std(hypothesis_burned):.2f}")
    
    # Calculate percentage change
    pct_change = ((np.mean(hypothesis_burned) - np.mean(control_burned)) / np.mean(control_burned)) * 100
    print(f"Average percentage change: {pct_change:.2f}%")
    
    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.hist(control_burned, bins=20, alpha=0.6, label='Control', color='blue')
    plt.hist(hypothesis_burned, bins=20, alpha=0.6, label='Hypothesis 2 (Dense Shrub +10%)', color='red')
    plt.xlabel('Final Burned Area (cells)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Burned Areas (100 Runs)')
    plt.legend()
    plt.savefig('hypothesis_2_histogram.png')
    plt.show()
    
    # Plot average burned area over time
    avg_control = np.mean([r['burned_over_time'] for r in control_results], axis=0)
    avg_hypothesis = np.mean([r['burned_over_time'] for r in hypothesis_results], axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(avg_control, label='Control', color='blue')
    plt.plot(avg_hypothesis, label='Hypothesis 2', color='red')
    plt.fill_between(range(len(avg_control)), 
                    avg_control - np.std([r['burned_over_time'] for r in control_results], axis=0),
                    avg_control + np.std([r['burned_over_time'] for r in control_results], axis=0),
                    alpha=0.2, color='blue')
    plt.fill_between(range(len(avg_hypothesis)),
                    avg_hypothesis - np.std([r['burned_over_time'] for r in hypothesis_results], axis=0),
                    avg_hypothesis + np.std([r['burned_over_time'] for r in hypothesis_results], axis=0),
                    alpha=0.2, color='red')
    plt.title('Average Burned Area Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Burned Area (cells)')
    plt.legend()
    plt.grid(True)
    plt.savefig('hypothesis_2_time_series.png')
    plt.show()

def main():
    # Run hypothesis 2 testing
    control_results, hypothesis_results = run_hypothesis_2_test(n_runs=100)
    
    # Analyze and visualize results
    analyze_results(control_results, hypothesis_results)

if __name__ == "__main__":
    main() 