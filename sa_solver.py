# pyrefly: ignore [missing-import]
import numpy as np
import random
import math
import time

from lns_solver import LNSSolver

class SimulatedAnnealingSolver:
    def __init__(self, 
                 distance_matrix: np.ndarray, 
                 initial_temp: float = 1000.0, 
                 alpha: float = 0.95, 
                 min_temp: float = 0.01,
                 lns_prob: float = 0.05):
        self.matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.initial_temp = initial_temp
        self.alpha = alpha
        self.min_temp = min_temp
        self.lns_prob = lns_prob

    def calculate_cost(self, tour: list) -> float:
        cost = 0.0
        for i in range(self.num_cities - 1):
            cost += self.matrix[tour[i], tour[i+1]]
        cost += self.matrix[tour[-1], tour[0]]
        return cost

    def _get_2opt_cost_diff(self, tour: list, i: int, j: int) -> float:
        c_prev = tour[i - 1]
        c_i = tour[i]
        c_j = tour[j]
        c_next = tour[(j + 1) % self.num_cities]
        
        old_cost = self.matrix[c_prev, c_i] + self.matrix[c_j, c_next]
        new_cost = self.matrix[c_prev, c_j] + self.matrix[c_i, c_next]
        
        return new_cost - old_cost

    def solve(self, steps_per_temp: int = 50, max_cooling_steps: int = 300) -> tuple:
        """
        Solves TSP using Simulated Annealing.
        Returns:
            best_tour: list of city indices representing the best route found
            best_cost: float representing the cost of the best route
            history: list of float representing the best cost found at each cooling step
        """
        # Start from a purely random initial route
        current_tour = list(range(self.num_cities))
        random.shuffle(current_tour)
        
        # Ensure city 0 is at index 0 (arbitrary choice, symmetric shifts are identical)
        idx_zero = current_tour.index(0)
        current_tour = current_tour[idx_zero:] + current_tour[:idx_zero]
        
        current_cost = self.calculate_cost(current_tour)
        
        global_best_tour = current_tour.copy()
        global_best_cost = current_cost
        
        temp = self.initial_temp
        history = []
        
        for step in range(max_cooling_steps):
            if temp < self.min_temp:
                break
                
            for _ in range(steps_per_temp):
                if self.lns_prob > 0.0 and random.random() < self.lns_prob:
                    # Generate candidate using LNS (1 iteration, destroy size 4 to 6)
                    lns = LNSSolver(self.matrix, min_destroy=4, max_destroy=6)
                    candidate, candidate_cost = lns.solve(current_tour, iterations=1)
                    diff = candidate_cost - current_cost
                    
                    if diff < 0 or random.random() < math.exp(-diff / temp):
                        current_tour = candidate
                        current_cost = candidate_cost
                        
                        if current_cost < global_best_cost:
                            global_best_cost = current_cost
                            global_best_tour = current_tour.copy()
                else:
                    # Generate a random 2-opt neighbor
                    # indices i and j: 1 <= i < j <= num_cities - 1
                    i, j = sorted(random.sample(range(1, self.num_cities), 2))
                    
                    diff = self._get_2opt_cost_diff(current_tour, i, j)
                    
                    # Metropolis criterion
                    if diff < 0 or random.random() < math.exp(-diff / temp):
                        # Accept the move
                        current_tour[i:j+1] = current_tour[i:j+1][::-1]
                        current_cost += diff
                        
                        # Update global best
                        if current_cost < global_best_cost:
                            global_best_cost = current_cost
                            global_best_tour = current_tour.copy()
            
            # Record current global best cost
            history.append(global_best_cost)
            
            # Cool down
            temp *= self.alpha
            
        return global_best_tour, global_best_cost, history

if __name__ == '__main__':
    # Quick test
    import os
    from ingestion import load_distance_matrix
    base_dir = os.path.dirname(os.path.abspath(__file__))
    m = load_distance_matrix(os.path.join(base_dir, 'berlin52-tsp.txt'))
    solver = SimulatedAnnealingSolver(m, initial_temp=1000.0, alpha=0.95, min_temp=0.01)
    start_t = time.time()
    best_tour, best_cost, history = solver.solve(steps_per_temp=100, max_cooling_steps=200)
    print("SA Run completed in:", time.time() - start_t)
    print("Best cost found:", best_cost)
    print("History length:", len(history))
