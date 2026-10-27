import pyomo.environ as pyo
import numpy as np
import time
import os
from ingestion import load_distance_matrix

class BILPSolver:
    def __init__(self, distance_matrix: np.ndarray, time_limit: float = 1800.0):
        self.matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.time_limit = time_limit

    def _nearest_neighbor(self) -> list:
        n = self.num_cities
        unvisited = set(range(1, n))
        tour = [0]
        current = 0
        while unvisited:
            next_city = min(unvisited, key=lambda c: self.matrix[current, c])
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        return tour

    def solve(self, log_file: str = "gurobi_solver_log.txt") -> tuple:
        """
        Solves symmetric TSP using Pyomo (BILP with MTZ formulation) and Gurobi.
        Returns:
            best_tour: list of city indices representing the optimal route
            best_cost: float representing the cost of the optimal route
            solve_time: float representing solver runtime in seconds
        """
        # Create Concrete Model
        model = pyo.ConcreteModel()
        
        # Set of vertices
        model.V = pyo.RangeSet(0, self.num_cities - 1)
        
        # Variables: x[i,j] is binary, u[i] is continuous ordering for MTZ
        model.x = pyo.Var(model.V, model.V, within=pyo.Binary)
        model.u = pyo.Var(model.V, within=pyo.NonNegativeReals, bounds=(1, self.num_cities - 1))
        
        # Fix diagonal x[i,i] to 0
        for i in model.V:
            model.x[i, i].fix(0.0)
            
        # Objective: minimize total distance
        def obj_rule(m):
            return sum(self.matrix[i, j] * m.x[i, j] for i in m.V for j in m.V if i != j)
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        
        # Constraints: leave each city exactly once
        def leave_rule(m, i):
            return sum(m.x[i, j] for j in m.V if i != j) == 1
        model.leave = pyo.Constraint(model.V, rule=leave_rule)
        
        # Constraints: enter each city exactly once
        def enter_rule(m, j):
            return sum(m.x[i, j] for i in m.V if i != j) == 1
        model.enter = pyo.Constraint(model.V, rule=enter_rule)
        
        # Heuristic 1: Eliminate 2-cycles directly (x_ij + x_ji <= 1)
        # Helps solver prune simple back-and-forth loops immediately
        def symmetric_rule(m, i, j):
            if i >= j:
                return pyo.Constraint.Skip
            return m.x[i, j] + m.x[j, i] <= 1
        model.symmetric = pyo.Constraint(model.V, model.V, rule=symmetric_rule)
        
        # Subtour Elimination: Miller-Tucker-Zemlin (MTZ) formulation
        # u_i - u_j + N * x_ij <= N - 1
        def mtz_rule(m, i, j):
            if i == 0 or j == 0 or i == j:
                return pyo.Constraint.Skip
            return m.u[i] - m.u[j] + self.num_cities * m.x[i, j] <= self.num_cities - 1
        model.mtz = pyo.Constraint(model.V, model.V, rule=mtz_rule)
        
        # Heuristic 2: Warm Start using Nearest Neighbor tour
        nn_tour = self._nearest_neighbor()
        tour_edges = set((nn_tour[k], nn_tour[(k+1) % self.num_cities]) for k in range(self.num_cities))
        
        # Warm start binary variables
        for i in model.V:
            for j in model.V:
                if i != j:
                    model.x[i, j].value = 1.0 if (i, j) in tour_edges else 0.0
                    
        # Warm start MTZ sequence variables
        for k in range(1, self.num_cities):
            model.u[nn_tour[k]].value = float(k)
            
        # Configure solver
        solver = pyo.SolverFactory('gurobi')
        solver.options['TimeLimit'] = self.time_limit
        solver.options['MIPFocus'] = 1  # Focus on finding high quality feasible solutions early
        if log_file:
            solver.options['LogFile'] = log_file
            
        t_start = time.time()
        results = solver.solve(model, warmstart=True, tee=True)
        solve_time = time.time() - t_start
        
        # Extract tour from binary values
        tour = [0]
        current = 0
        visited = {0}
        
        while len(tour) < self.num_cities:
            next_city = None
            for j in model.V:
                if j != current and j not in visited:
                    val = pyo.value(model.x[current, j])
                    if val is not None and val > 0.5:
                        next_city = j
                        break
            if next_city is None:
                # If solver didn't complete or returned fractional, fallback to NN path
                print("Warning: Solver output did not form a complete tour, using NN fallback.")
                return nn_tour, float(sum(self.matrix[nn_tour[k], nn_tour[(k+1)%self.num_cities]] for k in range(self.num_cities))), solve_time
            tour.append(next_city)
            visited.add(next_city)
            current = next_city
            
        optimal_cost = float(pyo.value(model.obj))
        return tour, optimal_cost, solve_time

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    m = load_distance_matrix(os.path.join(base_dir, 'berlin52-tsp.txt'))
    
    print("Testing Pyomo + Gurobi BILP Solver on Berlin52...")
    solver = BILPSolver(m, time_limit=30.0)
    tour, cost, dur = solver.solve("test_gurobi_solver_log.txt")
    print(f"Optimal Tour Cost: {cost}")
    print(f"Tour Path: {tour}")
    print(f"Solve Duration: {dur:.4f}s")
