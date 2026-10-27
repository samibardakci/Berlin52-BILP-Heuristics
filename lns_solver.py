import numpy as np
import random
import time
import os
from ingestion import load_distance_matrix

class LNSSolver:
    def __init__(self, distance_matrix: np.ndarray, min_destroy: int = 4, max_destroy: int = 10):
        self.matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.min_destroy = min_destroy
        self.max_destroy = max_destroy

    def calculate_cost(self, tour: list) -> float:
        cost = 0.0
        for i in range(self.num_cities - 1):
            cost += self.matrix[tour[i], tour[i+1]]
        cost += self.matrix[tour[-1], tour[0]]
        return cost

    def _destroy_random(self, tour: list, k: int) -> tuple:
        removed = random.sample(tour, k)
        partial_tour = [c for c in tour if c not in removed]
        return partial_tour, removed

    def _destroy_relatedness(self, tour: list, k: int) -> tuple:
        # Pick a random seed city
        seed = random.choice(tour)
        # Calculate distances to seed
        dists = [(c, self.matrix[seed, c]) for c in tour if c != seed]
        # Sort by distance
        dists.sort(key=lambda x: x[1])
        # Select k-1 closest to seed
        removed = [seed] + [c for c, _ in dists[:k-1]]
        partial_tour = [c for c in tour if c not in removed]
        return partial_tour, removed

    def _repair_cheapest_insertion(self, partial_tour: list, removed: list) -> list:
        tour = partial_tour.copy()
        rem = removed.copy()
        
        while rem:
            best_city = None
            best_pos = None
            best_increase = float('inf')
            
            # For each removed city, evaluate all insertion positions
            for city in rem:
                n = len(tour)
                for i in range(n):
                    c_curr = tour[i]
                    c_next = tour[(i + 1) % n]
                    
                    # Cost increase if inserted between c_curr and c_next
                    increase = self.matrix[c_curr, city] + self.matrix[city, c_next] - self.matrix[c_curr, c_next]
                    
                    if increase < best_increase:
                        best_increase = increase
                        best_city = city
                        best_pos = i + 1  # Insert after index i
                        
            # Insert the best city at its best position
            tour.insert(best_pos, best_city)
            rem.remove(best_city)
            
        return tour

    def solve(self, initial_tour: list, iterations: int = 150) -> tuple:
        """
        Polishes an initial tour using Large Neighborhood Search.
        Returns:
            best_tour: list representing improved tour path
            best_cost: float representing cost of improved tour
        """
        best_tour = initial_tour.copy()
        # Ensure city 0 is at index 0 (canonical representation)
        if best_tour[0] != 0:
            idx_zero = best_tour.index(0)
            best_tour = best_tour[idx_zero:] + best_tour[:idx_zero]
            
        best_cost = self.calculate_cost(best_tour)
        
        for _ in range(iterations):
            # Select destroy size
            k = random.randint(self.min_destroy, min(self.max_destroy, self.num_cities - 3))
            
            # Choose destroy method (70% relatedness, 30% random)
            if random.random() < 0.7:
                partial, removed = self._destroy_relatedness(best_tour, k)
            else:
                partial, removed = self._destroy_random(best_tour, k)
                
            # Repair
            candidate = self._repair_cheapest_insertion(partial, removed)
            candidate_cost = self.calculate_cost(candidate)
            
            # Accept if strictly better
            if candidate_cost < best_cost:
                best_tour = candidate
                best_cost = candidate_cost
                
        # Final canonical rotation
        if best_tour[0] != 0:
            idx_zero = best_tour.index(0)
            best_tour = best_tour[idx_zero:] + best_tour[:idx_zero]
            
        return best_tour, best_cost

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    m = load_distance_matrix(os.path.join(base_dir, 'berlin52-tsp.txt'))
    
    # Generate a random initial tour
    init = list(range(len(m)))
    random.shuffle(init)
    
    solver = LNSSolver(m)
    print("Initial tour cost:", solver.calculate_cost(init))
    
    t_start = time.time()
    best_t, best_c = solver.solve(init, iterations=500)
    
    print(f"LNS Solve completed in {time.time() - t_start:.4f}s")
    print("Best tour cost found:", best_c)
    print("Best tour path:", best_t)
