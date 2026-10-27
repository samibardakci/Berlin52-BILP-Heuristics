import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from orchestrator import TSPOrchestrator

def run_experiment():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tsp_path = os.path.join(base_dir, 'berlin52-tsp.txt')
    
    print("=" * 60)
    print("      Berlin52 Symmetric TSP Solver Comparison Framework")
    print("=" * 60)
    print(f"Dataset path: {tsp_path}")
    
    # Instantiate orchestrator
    orchestrator = TSPOrchestrator(tsp_path)
    
    # Custom parameter setup with LNS integrated
    config = {
        'GA': {
            'pop_size': 400,
            'generations': 5000,
            'crossover_rate': 0.70,
            'mutation_rate': 0.18,
            'tournament_size': 2,
            'elite_size': 80,
            'lns_rate': 0.12
        },
        'TS': {
            'iterations': 10000,
            'tabu_tenure': 18,
            'stagnation_limit': 150,
            'lns_intensify': True
        },
        'SA': {
            'max_cooling_steps': 5000,
            'steps_per_temp': 80,
            'initial_temp': 1700.0,
            'alpha': 0.99,
            'min_temp': 0.005,
            'lns_prob': 0.04
        }
    }
    
    t_start = time.time()
    results = orchestrator.run(config)
    
    # Run Pyomo + Gurobi BILP solver
    bilp_results = orchestrator.run_bilp(time_limit=60.0)
    
    total_duration = time.time() - t_start
    
    print("\n" + "=" * 60)
    print("                     Execution Summary")
    print("=" * 60)
    print(f"{'Algorithm':<15}{'Best Cost':<15}{'Time (s)':<15}{'History Length':<15}")
    print("-" * 60)
    
    for alg in ['GA', 'TS', 'SA']:
        cost = results[alg]['cost']
        duration = results[alg]['time']
        history_len = len(results[alg]['history'])
        print(f"{alg:<15}{cost:<15.2f}{duration:<15.4f}{history_len:<15}")
        
    print(f"{'BILP (Opt)':<15}{bilp_results['cost']:<15.2f}{bilp_results['time']:<15.4f}{'N/A (Exact)':<15}")
    print("-" * 60)
    print(f"Total pipeline run completed in {total_duration:.2f} seconds.")
    excel_path = os.path.join(base_dir, 'tsp_results.xlsx')
    print(f"Excel report successfully created at: {excel_path}")
    print(f"BILP Excel report successfully created at: {os.path.join(base_dir, 'tsp_bilp_results.xlsx')}")
    
    # 3. Create a seaborn convergence plot image
    create_seaborn_plot(results, bilp_results['cost'])

def create_seaborn_plot(results: dict, bilp_cost: float):
    print("\nGenerating convergence curves using seaborn...")
    
    # Setup seaborn styles
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    colors = {'GA': '#4285F4', 'TS': '#EA4335', 'SA': '#FBBC05'}
    
    for alg in ['GA', 'TS', 'SA']:
        history = results[alg]['history']
        iterations = np.arange(1, len(history) + 1)
        
        sns.lineplot(
            x=iterations, 
            y=history, 
            label=f"{alg} (Best: {results[alg]['cost']:.2f})", 
            color=colors[alg],
            linewidth=2
        )
        
    # Draw BILP exact solution reference line
    plt.axhline(
        y=bilp_cost, 
        color='#34A853', 
        linestyle='--', 
        linewidth=2.0, 
        label=f'Optimal (BILP): {bilp_cost:.2f}'
    )
        
    plt.title('Convergence Analysis of TSP Metaheuristics vs Exact BILP (Berlin52)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Algorithm Iteration / Generation / Cooling Step', fontsize=11, labelpad=10)
    plt.ylabel('Best Tour Distance / Cost', fontsize=11, labelpad=10)
    plt.legend(title="Algorithm", title_fontsize='11', loc='upper right', frameon=True)
    plt.tight_layout()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(base_dir, 'convergence_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Convergence plot saved to: {plot_path}")

if __name__ == '__main__':
    run_experiment()
