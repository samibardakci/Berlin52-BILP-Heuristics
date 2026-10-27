import numpy as np
import pandas as pd
import time
import os
import xlsxwriter

from ingestion import load_distance_matrix
from ga_solver import GASolver
from ts_solver import TabuSearchSolver
from sa_solver import SimulatedAnnealingSolver
from bilp_solver import BILPSolver

class TSPOrchestrator:
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path
        self.matrix = load_distance_matrix(data_file_path)
        self.num_cities = len(self.matrix)
        
        self.default_config = {
            'GA': {
                'pop_size': 2000,
                'generations': 500,
                'crossover_rate': 0.8,
                'mutation_rate': 0.25,
                'tournament_size': 5,
                'elite_size': 2,
                'lns_rate': 0.1
            },
            'TS': {
                'iterations': 30000,
                'tabu_tenure': 15,
                'stagnation_limit': 150,
                'lns_intensify': True
            },
            'SA': {
                'max_cooling_steps': 2500,
                'steps_per_temp': 60,
                'initial_temp': 1200.0,
                'alpha': 0.95,
                'min_temp': 0.01,
                'lns_prob': 0.05
            }
        }

    def run(self, custom_config: dict = None) -> dict:
        # Merge custom configs with defaults
        config = self.default_config.copy()
        if custom_config:
            for alg, params in custom_config.items():
                if alg in config:
                    config[alg].update(params)
                else:
                    config[alg] = params
                    
        results = {}
        
        # 1. Run Genetic Algorithm
        print("Starting Genetic Algorithm...")
        ga_conf = config['GA']
        ga = GASolver(
            distance_matrix=self.matrix,
            pop_size=ga_conf['pop_size'],
            crossover_rate=ga_conf['crossover_rate'],
            mutation_rate=ga_conf['mutation_rate'],
            tournament_size=ga_conf['tournament_size'],
            elite_size=ga_conf['elite_size'],
            lns_rate=ga_conf.get('lns_rate', 0.1)
        )
        t0 = time.time()
        ga_tour, ga_cost, ga_history = ga.solve(generations=ga_conf['generations'])
        ga_time = time.time() - t0
        print(f"GA finished. Best Cost: {ga_cost:.2f} in {ga_time:.4f}s")
        results['GA'] = {
            'tour': ga_tour,
            'cost': ga_cost,
            'time': ga_time,
            'history': ga_history,
            'config': ga_conf
        }
        
        # 2. Run Tabu Search
        print("Starting Tabu Search...")
        ts_conf = config['TS']
        ts = TabuSearchSolver(
            distance_matrix=self.matrix,
            tabu_tenure=ts_conf['tabu_tenure'],
            stagnation_limit=ts_conf.get('stagnation_limit', 150),
            lns_intensify=ts_conf.get('lns_intensify', True)
        )
        t0 = time.time()
        ts_tour, ts_cost, ts_history = ts.solve(iterations=ts_conf['iterations'])
        ts_time = time.time() - t0
        print(f"TS finished. Best Cost: {ts_cost:.2f} in {ts_time:.4f}s")
        results['TS'] = {
            'tour': ts_tour,
            'cost': ts_cost,
            'time': ts_time,
            'history': ts_history,
            'config': ts_conf
        }
        
        # 3. Run Simulated Annealing
        print("Starting Simulated Annealing...")
        sa_conf = config['SA']
        sa = SimulatedAnnealingSolver(
            distance_matrix=self.matrix,
            initial_temp=sa_conf['initial_temp'],
            alpha=sa_conf['alpha'],
            min_temp=sa_conf['min_temp'],
            lns_prob=sa_conf.get('lns_prob', 0.05)
        )
        t0 = time.time()
        sa_tour, sa_cost, sa_history = sa.solve(
            steps_per_temp=sa_conf['steps_per_temp'],
            max_cooling_steps=sa_conf['max_cooling_steps']
        )
        sa_time = time.time() - t0
        print(f"SA finished. Best Cost: {sa_cost:.2f} in {sa_time:.4f}s")
        results['SA'] = {
            'tour': sa_tour,
            'cost': sa_cost,
            'time': sa_time,
            'history': sa_history,
            'config': sa_conf
        }
        
        # Save results to Excel
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(base_dir, 'tsp_results.xlsx')
        self._write_excel(results, output_file)
        return results

    def _write_excel(self, results: dict, output_file: str):
        print(f"Generating Excel report: {output_file}")
        
        workbook = xlsxwriter.Workbook(output_file)
        
        # Style definitions
        font_name = 'Segoe UI'
        
        title_format = workbook.add_format({
            'bold': True, 'font_size': 16, 'font_name': font_name,
            'font_color': '#FFFFFF', 'bg_color': '#1B365D',
            'align': 'center', 'valign': 'vcenter'
        })
        
        subtitle_format = workbook.add_format({
            'font_size': 11, 'font_name': font_name,
            'italic': True, 'font_color': '#555555',
            'align': 'left', 'valign': 'vcenter'
        })
        
        header_format = workbook.add_format({
            'bold': True, 'font_name': font_name, 'font_size': 11,
            'font_color': '#FFFFFF', 'bg_color': '#2C4A73',
            'align': 'center', 'valign': 'vcenter', 'border': 1, 'border_color': '#D0D5DD'
        })
        
        data_format = workbook.add_format({
            'font_name': font_name, 'font_size': 10,
            'align': 'center', 'valign': 'vcenter', 'border': 1, 'border_color': '#E4E7EC'
        })
        
        num_format = workbook.add_format({
            'font_name': font_name, 'font_size': 10, 'num_format': '#,##0.00',
            'align': 'right', 'valign': 'vcenter', 'border': 1, 'border_color': '#E4E7EC'
        })
        
        num_int_format = workbook.add_format({
            'font_name': font_name, 'font_size': 10, 'num_format': '#,##0',
            'align': 'center', 'valign': 'vcenter', 'border': 1, 'border_color': '#E4E7EC'
        })
        
        best_num_format = workbook.add_format({
            'bold': True, 'font_name': font_name, 'font_size': 10, 'num_format': '#,##0.00',
            'font_color': '#15803D', 'bg_color': '#DCFCE7', # soft green
            'align': 'right', 'valign': 'vcenter', 'border': 1, 'border_color': '#BBF7D0'
        })

        best_text_format = workbook.add_format({
            'bold': True, 'font_name': font_name, 'font_size': 10,
            'font_color': '#15803D', 'bg_color': '#DCFCE7',
            'align': 'center', 'valign': 'vcenter', 'border': 1, 'border_color': '#BBF7D0'
        })
        
        config_title_format = workbook.add_format({
            'bold': True, 'font_name': font_name, 'font_size': 11,
            'bg_color': '#F2F4F7', 'font_color': '#1B365D',
            'align': 'left', 'valign': 'vcenter', 'border': 1, 'border_color': '#D0D5DD'
        })

        # -------------------------------------------------------------------------
        # SHEET 1: Summary Dashboard
        # -------------------------------------------------------------------------
        summary_sheet = workbook.add_worksheet('Summary')
        summary_sheet.set_zoom(100)
        summary_sheet.hide_gridlines(2) # Show gridlines when printed, but screen gridlines are kept or customized
        
        # Set column widths
        summary_sheet.set_column('A:A', 22)
        summary_sheet.set_column('B:D', 16)
        summary_sheet.set_column('E:L', 12)
        
        # Title Block
        summary_sheet.merge_range('A1:D2', 'TSP SOLVER SUITE DASHBOARD', title_format)
        summary_sheet.write('A3', 'Dataset: Berlin52 Symmetric TSP (52 Locations)', subtitle_format)
        
        # Summary Table Headers
        summary_sheet.write('A5', 'Algorithm', header_format)
        summary_sheet.write('B5', 'Best Tour Cost', header_format)
        summary_sheet.write('C5', 'Execution Time (s)', header_format)
        summary_sheet.write('D5', 'Iterations / Gens', header_format)
        
        algorithms = ['GA', 'TS', 'SA']
        best_alg = min(algorithms, key=lambda a: results[a]['cost'])
        
        for idx, alg in enumerate(algorithms):
            row_idx = 5 + idx
            res = results[alg]
            
            # Use green highlight for the best algorithm
            is_best = (alg == best_alg)
            text_fmt = best_text_format if is_best else data_format
            cost_fmt = best_num_format if is_best else num_format
            
            summary_sheet.write(row_idx, 0, alg, text_fmt)
            summary_sheet.write(row_idx, 1, res['cost'], cost_fmt)
            summary_sheet.write(row_idx, 2, res['time'], num_format)
            
            # Step count representation
            steps = len(res['history'])
            summary_sheet.write(row_idx, 3, steps, num_int_format)
            
        # Metadata / Configurations table
        summary_sheet.write('A10', 'Solver Parameter Configurations', config_title_format)
        summary_sheet.merge_range('B10:D10', '', config_title_format)
        
        param_headers = ['Algorithm', 'Parameter', 'Value', '']
        for col_idx, text in enumerate(param_headers[:3]):
            summary_sheet.write(10, col_idx, text, header_format)
            
        config_rows = []
        for alg in algorithms:
            for param, val in results[alg]['config'].items():
                config_rows.append((alg, param, val))
                
        for idx, (alg, param, val) in enumerate(config_rows):
            r = 11 + idx
            summary_sheet.write(r, 0, alg, data_format)
            summary_sheet.write(r, 1, param, data_format)
            if isinstance(val, float):
                summary_sheet.write(r, 2, val, num_format)
            else:
                summary_sheet.write(r, 2, val, num_int_format)
                
        # -------------------------------------------------------------------------
        # Helper Sheet: Convergence_Data (to feed the line chart)
        # -------------------------------------------------------------------------
        data_sheet = workbook.add_worksheet('Convergence_Data')
        data_sheet.write(0, 0, 'Iteration')
        data_sheet.write(0, 1, 'GA')
        data_sheet.write(0, 2, 'TS')
        data_sheet.write(0, 3, 'SA')
        
        ga_hist = results['GA']['history']
        ts_hist = results['TS']['history']
        sa_hist = results['SA']['history']
        
        max_len = max(len(ga_hist), len(ts_hist), len(sa_hist))
        
        for i in range(max_len):
            r = i + 1
            data_sheet.write(r, 0, r)
            
            # GA value (pad with last value if finished)
            if i < len(ga_hist):
                data_sheet.write(r, 1, ga_hist[i])
            else:
                data_sheet.write(r, 1, ga_hist[-1])
                
            # TS value
            if i < len(ts_hist):
                data_sheet.write(r, 2, ts_hist[i])
            else:
                data_sheet.write(r, 2, ts_hist[-1])
                
            # SA value
            if i < len(sa_hist):
                data_sheet.write(r, 3, sa_hist[i])
            else:
                data_sheet.write(r, 3, sa_hist[-1])

        # Create convergence line chart
        line_chart = workbook.add_chart({'type': 'line'})
        line_chart.set_title({'name': 'Algorithm Convergence Profile (Symmetric TSP)'})
        line_chart.set_x_axis({'name': 'Iteration / Step'})
        line_chart.set_y_axis({'name': 'Best Tour Cost (units)'})
        
        line_chart.add_series({
            'name': '=Convergence_Data!$B$1',
            'categories': f'=Convergence_Data!$A$2:$A${max_len+1}',
            'values': f'=Convergence_Data!$B$2:$B${max_len+1}',
            'line': {'color': '#4285F4', 'width': 2.0} # Google Blue
        })
        line_chart.add_series({
            'name': '=Convergence_Data!$C$1',
            'categories': f'=Convergence_Data!$A$2:$A${max_len+1}',
            'values': f'=Convergence_Data!$C$2:$C${max_len+1}',
            'line': {'color': '#EA4335', 'width': 2.0} # Google Red
        })
        line_chart.add_series({
            'name': '=Convergence_Data!$D$1',
            'categories': f'=Convergence_Data!$A$2:$A${max_len+1}',
            'values': f'=Convergence_Data!$D$2:$D${max_len+1}',
            'line': {'color': '#FBBC05', 'width': 2.0} # Google Yellow
        })
        
        # Position convergence chart on Summary sheet
        summary_sheet.insert_chart('F4', line_chart, {'x_scale': 1.5, 'y_scale': 1.2})
        
        # Create runtime pie chart
        pie_chart = workbook.add_chart({'type': 'pie'})
        pie_chart.set_title({'name': 'Execution Time Share (%)'})
        pie_chart.add_series({
            'name': 'Runtime Allocation',
            'categories': '=Summary!$A$6:$A$8',
            'values': '=Summary!$C$6:$C$8',
            'points': [
                {'fill': {'color': '#4285F4'}}, # GA = Blue
                {'fill': {'color': '#EA4335'}}, # TS = Red
                {'fill': {'color': '#FBBC05'}}  # SA = Yellow
            ]
        })
        
        summary_sheet.insert_chart('F20', pie_chart, {'x_scale': 1.2, 'y_scale': 1.0})

        # -------------------------------------------------------------------------
        # SHEETS for Individual Algorithms: GA_Results, TS_Results, SA_Results
        # -------------------------------------------------------------------------
        for alg in algorithms:
            sheet_name = f'{alg}_Results'
            sh = workbook.add_worksheet(sheet_name)
            res = results[alg]
            
            sh.set_column('A:A', 15)
            sh.set_column('B:B', 15)
            sh.set_column('D:D', 15)
            sh.set_column('E:E', 15)
            
            # Sheet Title
            sh.merge_range('A1:B2', f'{alg} Solve Results', title_format)
            
            # Subtitle
            sh.write('A3', f'Optimal Route Cost: {res["cost"]:.2f} | Time: {res["time"]:.4f}s', subtitle_format)
            
            # Route tour path sequence
            sh.write('A5', 'Route Position', header_format)
            sh.write('B5', 'City ID (0-51)', header_format)
            
            tour = res['tour']
            for pos, city in enumerate(tour):
                row = 5 + pos
                sh.write(row, 0, pos, num_int_format)
                sh.write(row, 1, city, num_int_format)
                
            # Convergence trace data
            sh.write('D5', 'Iteration/Gen', header_format)
            sh.write('E5', 'Best Cost Found', header_format)
            
            history_data = res['history']
            for step_idx, cost_val in enumerate(history_data):
                row = 5 + step_idx
                sh.write(row, 3, step_idx + 1, num_int_format)
                sh.write(row, 4, cost_val, num_format)
                
        workbook.close()
        print("Excel report successfully created.")

    def run_bilp(self, time_limit: float = 60.0) -> dict:
        print("Starting Pyomo + Gurobi BILP Solver with MTZ formulation...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_file = os.path.join(base_dir, "gurobi_solver_log.txt")
        
        solver = BILPSolver(self.matrix, time_limit=time_limit)
        tour, cost, duration = solver.solve(log_file=log_file)
        
        print(f"BILP Solver finished. Optimal Cost: {cost:.2f} in {duration:.4f}s")
        print(f"Gurobi solver log saved to: {log_file}")
        
        results = {
            'tour': tour,
            'cost': cost,
            'time': duration,
            'time_limit': time_limit
        }
        
        output_file = os.path.join(base_dir, 'tsp_bilp_results.xlsx')
        self._write_bilp_excel(results, output_file)
        return results

    def _write_bilp_excel(self, results: dict, output_file: str):
        print(f"Generating BILP Excel report: {output_file}")
        workbook = xlsxwriter.Workbook(output_file)
        font_name = 'Segoe UI'
        
        title_format = workbook.add_format({
            'bold': True, 'font_size': 16, 'font_name': font_name,
            'font_color': '#FFFFFF', 'bg_color': '#1E3A8A', # deep blue
            'align': 'center', 'valign': 'vcenter'
        })
        
        subtitle_format = workbook.add_format({
            'font_size': 11, 'font_name': font_name,
            'italic': True, 'font_color': '#4B5563',
            'align': 'left', 'valign': 'vcenter'
        })
        
        header_format = workbook.add_format({
            'bold': True, 'font_name': font_name, 'font_size': 11,
            'font_color': '#FFFFFF', 'bg_color': '#2563EB', # blue
            'align': 'center', 'valign': 'vcenter', 'border': 1, 'border_color': '#D1D5DB'
        })
        
        data_format = workbook.add_format({
            'font_name': font_name, 'font_size': 10,
            'align': 'center', 'valign': 'vcenter', 'border': 1, 'border_color': '#E5E7EB'
        })
        
        num_format = workbook.add_format({
            'font_name': font_name, 'font_size': 10, 'num_format': '#,##0.00',
            'align': 'right', 'valign': 'vcenter', 'border': 1, 'border_color': '#E5E7EB'
        })
        
        num_int_format = workbook.add_format({
            'font_name': font_name, 'font_size': 10, 'num_format': '#,##0',
            'align': 'center', 'valign': 'vcenter', 'border': 1, 'border_color': '#E5E7EB'
        })
        
        highlight_format = workbook.add_format({
            'bold': True, 'font_name': font_name, 'font_size': 10,
            'font_color': '#1E3A8A', 'bg_color': '#DBEAFE',
            'align': 'center', 'valign': 'vcenter', 'border': 1, 'border_color': '#BFDBFE'
        })
        
        highlight_num_format = workbook.add_format({
            'bold': True, 'font_name': font_name, 'font_size': 10, 'num_format': '#,##0.00',
            'font_color': '#1E3A8A', 'bg_color': '#DBEAFE',
            'align': 'right', 'valign': 'vcenter', 'border': 1, 'border_color': '#BFDBFE'
        })

        sheet = workbook.add_worksheet('BILP_Results')
        sheet.set_zoom(100)
        sheet.hide_gridlines(2)
        
        sheet.set_column('A:A', 24)
        sheet.set_column('B:B', 16)
        sheet.set_column('C:C', 4)
        sheet.set_column('D:D', 18)
        sheet.set_column('E:E', 22)
        
        # Title Block
        sheet.merge_range('A1:E2', 'TSP BILP SOLVER REPORT', title_format)
        sheet.write('A3', 'Solver backend: Pyomo + Gurobi (MTZ Formulation)', subtitle_format)
        
        # Performance Table
        sheet.write('A5', 'Metric', header_format)
        sheet.write('B5', 'Value', header_format)
        
        metrics = [
            ('Best Cost Found', results['cost'], highlight_num_format),
            ('Solve Time (seconds)', results['time'], num_format),
            ('Solver Time Limit (s)', results['time_limit'], num_int_format),
            ('Number of Cities', self.num_cities, num_int_format)
        ]
        
        for idx, (metric, val, fmt) in enumerate(metrics):
            row = 5 + idx
            sheet.write(row, 0, metric, data_format)
            sheet.write(row, 1, val, fmt)
            
        # Route Sequence Table
        sheet.write('D5', 'Route Position', header_format)
        sheet.write('E5', 'City ID (0-indexed)', header_format)
        
        tour = results['tour']
        for pos, city in enumerate(tour):
            row = 5 + pos
            sheet.write(row, 3, pos, num_int_format)
            sheet.write(row, 4, city, num_int_format)
            
        workbook.close()
        print("BILP Excel report successfully created.")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    orchestrator = TSPOrchestrator(os.path.join(base_dir, 'berlin52-tsp.txt'))
    # Use a small test run
    custom = {
        'GA': {'generations': 50, 'pop_size': 30},
        'TS': {'iterations': 50},
        'SA': {'max_cooling_steps': 50, 'steps_per_temp': 10}
    }
    orchestrator.run(custom)
    
    # Also test the BILP solver
    orchestrator.run_bilp(time_limit=30.0)

