# pyrefly: ignore [missing-import]
import numpy as np
import random
import time

from lns_solver import LNSSolver

class TabuSearchSolver:
    def __init__(self, distance_matrix: np.ndarray, tabu_tenure: int = 15, stagnation_limit: int = 50, lns_intensify: bool = True):
        self.matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.tabu_tenure = tabu_tenure
        self.stagnation_limit = stagnation_limit
        self.lns_intensify = lns_intensify

    def calculate_cost(self, tour: list) -> float:
        cost = 0.0
        for i in range(self.num_cities - 1):
            cost += self.matrix[tour[i], tour[i+1]]
        cost += self.matrix[tour[-1], tour[0]]
        return cost

    def _get_2opt_cost_diff(self, tour: list, i: int, j: int) -> float:
        # Assumes 1 <= i < j <= num_cities - 1
        c_prev = tour[i - 1]
        c_i = tour[i]
        c_j = tour[j]
        c_next = tour[(j + 1) % self.num_cities]
        
        # Cost of broken edges
        old_cost = self.matrix[c_prev, c_i] + self.matrix[c_j, c_next]
        # Cost of new edges
        new_cost = self.matrix[c_prev, c_j] + self.matrix[c_i, c_next]
        
        return new_cost - old_cost

    def _get_neighbors(self, tour: list):
        # We generate all possible 2-opt moves in indices 1 to num_cities - 1
        # (Fixing index 0 avoids cycle duplicates and simplifies border cases)
        for i in range(1, self.num_cities - 1):
            for j in range(i + 1, self.num_cities):
                yield i, j

    def solve(self, iterations: int = 300) -> tuple:
        """
        Solves TSP using Tabu Search.
        Returns:
            best_tour: list of city indices representing the best route found
            best_cost: float representing the cost of the best route
            history: list of float representing the best cost found at each iteration
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
        
        # Tabu list: maps (city_A, city_B) -> iteration until which it is tabu
        # We store (min(c1, c2), max(c1, c2)) to handle symmetry
        tabu_list = {}
        history = []
        no_improvement_count = 0
        
        for step in range(iterations):
            best_neighbor_move = None
            best_neighbor_tour = None
            best_neighbor_cost = float('inf')
            
            # Evaluate all 2-opt moves in the neighborhood
            for i, j in self._get_neighbors(current_tour):
                diff = self._get_2opt_cost_diff(current_tour, i, j)
                neighbor_cost = current_cost + diff
                
                c_i = current_tour[i]
                c_j = current_tour[j]
                move_pair = (min(c_i, c_j), max(c_i, c_j))
                
                is_tabu = step < tabu_list.get(move_pair, 0)
                # Aspiration criterion: if the move leads to a new global best, override tabu
                if is_tabu and neighbor_cost < global_best_cost:
                    is_tabu = False
                
                if not is_tabu:
                    if neighbor_cost < best_neighbor_cost:
                        best_neighbor_cost = neighbor_cost
                        best_neighbor_move = (i, j, move_pair)
            
            # If a valid neighbor move is found, apply it
            if best_neighbor_move is not None:
                i, j, move_pair = best_neighbor_move
                
                # Apply 2-opt reversal
                current_tour[i:j+1] = current_tour[i:j+1][::-1]
                current_cost = best_neighbor_cost
                
                # Update tabu list
                tabu_list[move_pair] = step + self.tabu_tenure
                
                # Update global best if improved
                if current_cost < global_best_cost:
                    global_best_cost = current_cost
                    global_best_tour = current_tour.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                # Fallback: if all moves are tabu and no aspiration occurs (rare),
                # just terminate or shuffle to escape (here we break to prevent freezing)
                break
                
            if self.lns_intensify and no_improvement_count >= self.stagnation_limit:
                lns = LNSSolver(self.matrix)
                polished_tour, polished_cost = lns.solve(global_best_tour, iterations=25)
                if polished_cost < global_best_cost:
                    global_best_cost = polished_cost
                    global_best_tour = polished_tour.copy()
                    current_tour = polished_tour.copy()
                    current_cost = polished_cost
                no_improvement_count = 0
                
            history.append(global_best_cost)
            
        return global_best_tour, global_best_cost, history

if __name__ == '__main__':
    # Quick test
    import os
    from ingestion import load_distance_matrix
    base_dir = os.path.dirname(os.path.abspath(__file__))
    m = load_distance_matrix(os.path.join(base_dir, 'berlin52-tsp.txt'))
    solver = TabuSearchSolver(m, tabu_tenure=15)
    start_t = time.time()
    best_tour, best_cost, history = solver.solve(iterations=200)
    print("TS Run completed in:", time.time() - start_t)
    print("Best cost found:", best_cost)
    print("History length:", len(history))
