# pyrefly: ignore [missing-import]
import numpy as np
import random
import time

from lns_solver import LNSSolver

class GASolver:
    def __init__(self, 
                 distance_matrix: np.ndarray, 
                 pop_size: int = 100, 
                 crossover_rate: float = 0.8, 
                 mutation_rate: float = 0.20, 
                 tournament_size: int = 5, 
                 elite_size: int = 2,
                 lns_rate: float = 0.1):
        self.matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.lns_rate = lns_rate

    def calculate_cost(self, tour: list) -> float:
        cost = 0.0
        for i in range(self.num_cities - 1):
            cost += self.matrix[tour[i], tour[i+1]]
        cost += self.matrix[tour[-1], tour[0]]  # Return to start city
        return cost

    def _create_initial_population(self) -> list:
        # Purely random initial routes
        population = []
        base = list(range(self.num_cities))
        for _ in range(self.pop_size):
            ind = base.copy()
            random.shuffle(ind)
            population.append(ind)
        return population

    def _tournament_selection(self, population: list, costs: list) -> list:
        # Select tournament_size random candidates and return the best one
        best_candidate_idx = random.choice(range(self.pop_size))
        for _ in range(self.tournament_size - 1):
            cand_idx = random.choice(range(self.pop_size))
            if costs[cand_idx] < costs[best_candidate_idx]:
                best_candidate_idx = cand_idx
        return population[best_candidate_idx].copy()

    def _ordered_crossover(self, parent1: list, parent2: list) -> list:
        # Ordered Crossover (OX)
        c1, c2 = sorted(random.sample(range(self.num_cities), 2))
        offspring = [-1] * self.num_cities
        
        # Copy the slice from parent1
        offspring[c1:c2+1] = parent1[c1:c2+1]
        copied_set = set(parent1[c1:c2+1])
        
        # Fill remaining positions from parent2 wrapping around
        p2_idx = (c2 + 1) % self.num_cities
        off_idx = (c2 + 1) % self.num_cities
        
        while -1 in offspring:
            val = parent2[p2_idx]
            if val not in copied_set:
                offspring[off_idx] = val
                copied_set.add(val)
                off_idx = (off_idx + 1) % self.num_cities
            p2_idx = (p2_idx + 1) % self.num_cities
            
        return offspring

    def _inversion_mutation(self, individual: list) -> list:
        # Inversion Mutation (2-opt style)
        i, j = sorted(random.sample(range(self.num_cities), 2))
        mutated = individual.copy()
        mutated[i:j+1] = mutated[i:j+1][::-1]
        return mutated

    def solve(self, generations: int = 500) -> tuple:
        """
        Solves TSP using Genetic Algorithm.
        Returns:
            best_tour: list of city indices representing the best route found
            best_cost: float representing the cost of the best route
            history: list of float representing the best cost found at each generation
        """
        # 1. Initialize population
        population = self._create_initial_population()
        history = []
        
        global_best_tour = None
        global_best_cost = float('inf')
        
        for gen in range(generations):
            # Calculate cost (fitness) for each individual
            costs = [self.calculate_cost(ind) for ind in population]
            
            # Find best individual in current generation
            best_gen_idx = np.argmin(costs)
            best_gen_cost = costs[best_gen_idx]
            best_gen_tour = population[best_gen_idx]
            
            if best_gen_cost < global_best_cost:
                global_best_cost = best_gen_cost
                global_best_tour = best_gen_tour.copy()
                
            # Apply LNS to polish the best individual found so far (Lamarckian Search)
            if self.lns_rate > 0.0 and random.random() < self.lns_rate:
                lns = LNSSolver(self.matrix)
                polished_tour, polished_cost = lns.solve(global_best_tour, iterations=15)
                if polished_cost < global_best_cost:
                    global_best_cost = polished_cost
                    global_best_tour = polished_tour.copy()
                    
            history.append(global_best_cost)
            
            # Elitism: carry over the best individuals directly
            sorted_indices = np.argsort(costs)
            next_population = [population[idx].copy() for idx in sorted_indices[:self.elite_size]]
            if len(next_population) > 0 and global_best_tour is not None:
                next_population[0] = global_best_tour.copy()
            
            # Generate offspring to fill the rest of the population
            while len(next_population) < self.pop_size:
                # Selection
                parent1 = self._tournament_selection(population, costs)
                parent2 = self._tournament_selection(population, costs)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring = self._ordered_crossover(parent1, parent2)
                else:
                    offspring = parent1.copy()
                    
                # Mutation
                if random.random() < self.mutation_rate:
                    offspring = self._inversion_mutation(offspring)
                    
                next_population.append(offspring)
                
            population = next_population
            
        return global_best_tour, global_best_cost, history

if __name__ == '__main__':
    # Quick test
    import os
    from ingestion import load_distance_matrix
    base_dir = os.path.dirname(os.path.abspath(__file__))
    m = load_distance_matrix(os.path.join(base_dir, 'berlin52-tsp.txt'))
    solver = GASolver(m, pop_size=100, crossover_rate=0.8, mutation_rate=0.2, elite_size=2)
    start_t = time.time()
    best_tour, best_cost, history = solver.solve(generations=200)
    print("GA Run completed in:", time.time() - start_t)
    print("Best cost found:", best_cost)
    print("History length:", len(history))
