
import re
import time
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from ortools.sat.python import cp_model

# ==========================================
# PART 1: Parser Logic (from create_wsp_parser.py)
# ==========================================

class Instance:
    def __init__(self, num_steps, num_users, num_constraints):
        self.num_steps = num_steps
        self.num_users = num_users
        self.num_constraints = num_constraints
        
        # User ID -> list of authorized Step IDs
        # Key presence implies explicit mention in file.
        self.authorizations = {} 
        
        # List of (step1, step2) tuples
        self.separation_duty = []
        
        # List of (step1, step2) tuples
        self.binding_duty = []
        
        # List of (k, [step_ids]) tuples
        self.at_most_k = []
        
        # List of (step_ids, [team1_users, team2_users, ...]) tuples
        self.one_team = []

    def __repr__(self):
        return (f"Instance(Steps={self.num_steps}, Users={self.num_users}, "
                f"Constraints={self.num_constraints}, "
                f"Auths={len(self.authorizations)}, "
                f"SoD={len(self.separation_duty)}, "
                f"BoD={len(self.binding_duty)}, "
                f"AMK={len(self.at_most_k)}, "
                f"OneTeam={len(self.one_team)})")

def read_file(filename):
    """
    Parses the WSP input file and returns an Instance object.
    Strictly follows implicit permission rules:
    - Users mentioned in Authorisations are restricted to those steps.
    - Users NOT mentioned are authorized for ALL steps.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    instance = None
    
    # Single line header pattern
    header_pattern = re.compile(r'#Steps:\s*(\d+),\s*#Users:\s*(\d+),\s*#Constraints:\s*(\d+)', re.IGNORECASE)
    # Multi-line patterns
    steps_pattern = re.compile(r'#Steps:\s*(\d+)', re.IGNORECASE)
    users_pattern = re.compile(r'#Users:\s*(\d+)', re.IGNORECASE)
    const_pattern = re.compile(r'#Constraints:\s*(\d+)', re.IGNORECASE)
    
    n_steps = None
    n_users = None
    n_const = None
    
    # helper
    def parse_ids(text, prefix):
        if not text:
            return []
        pattern = fr'{prefix}(\d+)'
        return [int(val) - 1 for val in re.findall(pattern, text, re.IGNORECASE)]

    # Line type patterns
    auth_pattern = re.compile(r'^authori[sz]ations\s+u(\d+)\s*(.*)', re.IGNORECASE)
    sod_pattern = re.compile(r'^separation-of-duty\s+s(\d+)\s+s(\d+)', re.IGNORECASE)
    bod_pattern = re.compile(r'^binding-of-duty\s+s(\d+)\s+s(\d+)', re.IGNORECASE)
    amk_pattern = re.compile(r'^at-most-k\s+(\d+)\s+(.*)', re.IGNORECASE)
    ot_pattern = re.compile(r'^one-team\s+(.*)', re.IGNORECASE)
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
            
        # Parse Header(s)
        if instance is None:
            # Try single line match
            match = header_pattern.match(line)
            if match:
                n_steps = int(match.group(1))
                n_users = int(match.group(2))
                n_const = int(match.group(3))
                instance = Instance(n_steps, n_users, n_const)
                continue
            
            # Try partial matches
            if n_steps is None:
                m = steps_pattern.match(line)
                if m: n_steps = int(m.group(1))
            if n_users is None:
                m = users_pattern.match(line)
                if m: n_users = int(m.group(1))
            if n_const is None:
                m = const_pattern.match(line)
                if m: n_const = int(m.group(1))
            
            # Check if we have all needed to create instance
            if n_steps is not None and n_users is not None and n_const is not None:
                instance = Instance(n_steps, n_users, n_const)
            
            # If we just found a header component, skip to next line to avoid misparsing
            if steps_pattern.match(line) or users_pattern.match(line) or const_pattern.match(line):
                continue
        
        if instance is None:
             continue

        # 1. Authorizations
        match = auth_pattern.match(line)
        if match:
            u_id = int(match.group(1)) - 1
            remainder = match.group(2)
            steps = parse_ids(remainder, 's')
            
            if u_id not in instance.authorizations:
                instance.authorizations[u_id] = []
            instance.authorizations[u_id].extend(steps)
            continue
            
        # 2. Separation-of-duty
        match = sod_pattern.match(line)
        if match:
            s1 = int(match.group(1)) - 1
            s2 = int(match.group(2)) - 1
            instance.separation_duty.append((s1, s2))
            continue
            
        # 3. Binding-of-duty
        match = bod_pattern.match(line)
        if match:
            s1 = int(match.group(1)) - 1
            s2 = int(match.group(2)) - 1
            instance.binding_duty.append((s1, s2))
            continue
            
        # 4. At-most-k
        match = amk_pattern.match(line)
        if match:
            k = int(match.group(1))
            remainder = match.group(2)
            steps = parse_ids(remainder, 's')
            instance.at_most_k.append((k, steps))
            continue
            
        # 5. One-team
        match = ot_pattern.match(line)
        if match:
            content = match.group(1)
            parts = content.split('(')
            steps_part = parts[0]
            steps = parse_ids(steps_part, 's')
            teams = []
            for team_part in parts[1:]:
                if ')' in team_part:
                    users_str = team_part.split(')')[0]
                    team_users = parse_ids(users_str, 'u')
                    teams.append(team_users)
            instance.one_team.append((steps, teams))
            continue
            
    return instance

# ==========================================
# PART 2: Solver Logic (from create_wsp_solver.py)
# ==========================================

class SolutionCallback(cp_model.CpSolverSolutionCallback):
    """Callback to monitor solver progress"""
    def __init__(self, progress_callback=None):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.progress_callback = progress_callback
        self.solution_count = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
    def on_solution_callback(self):
        """Called when a new solution is found"""
        self.solution_count += 1
        elapsed = time.time() - self.start_time
        
        if self.progress_callback:
            # Try to get objective value, but WSP is usually a satisfaction problem (no objective)
            objective = None
            try:
                objective = self.ObjectiveValue()
            except:
                pass  # No objective function, which is normal for WSP
            
            info = {
                'solutions': self.solution_count,
                'time': elapsed,
                'objective': objective,
                'status': f'Found solution #{self.solution_count}'
            }
            self.progress_callback(info)
            self.last_update_time = time.time()
    
    def get_stats(self):
        """Get current solver statistics"""
        elapsed = time.time() - self.start_time
        return {
            'solutions': self.solution_count,
            'time': elapsed,
            'status': 'Searching...' if self.solution_count == 0 else f'Found {self.solution_count} solution(s)'
        }

def Solver(filename, progress_callback=None):
    """
    Smart WSP Solver - Automatically chooses best algorithm
    
    Strategy:
    - Small instances (<300 users): Use CP-SAT
    - Large instances (>=300 users): Use Smart Backtracking with constraint propagation
    """
    import time
    
    start_time = time.time()
    
    # Quick parse to determine size
    try:
        instance = read_file(filename)
    except Exception as e:
        return {'sat': 'error', 'sol': [], 'mul_sol': f'Parse error: {e}', 'exe_time': '0ms'}
    
    # Choose algorithm based on instance size
    if instance.num_users >= 300:
        # Large instance: Use Smart Backtracking
        if progress_callback:
            progress_callback({'status': 'Using Smart Backtracking...', 'time': 0})
        
        try:
            from solver_backtrack import Solver_SmartBacktrack
            result = Solver_SmartBacktrack(filename, progress_callback)
            
            # If backtrack fails or times out, try Z3 as fallback
            if result['sat'] != 'sat' and 'z3' not in result['exe_time']:
                try:
                    if progress_callback:
                        progress_callback({'status': 'Backtrack failed, trying Z3...', 'time': time.time() - start_time})
                    
                    from solver_z3 import Solver_Z3
                    return Solver_Z3(filename, progress_callback)
                except:
                    pass
            
            return result
            
        except Exception as e:
            # Backtracking failed, fallback to CP-SAT
            if progress_callback:
                progress_callback({'status': f'Error: {e}, using CP-SAT...', 'time': time.time() - start_time})
    
    # Small/medium instance: Use CP-SAT
    return Solver_CPSAT(filename, progress_callback)

def Solver_CPSAT(filename, progress_callback=None):
    """
    Original CP-SAT based solver (renamed from Solver)
    """
    start_time = time.time()
    
    # Parse instance
    try:
        instance = read_file(filename)
    except Exception as e:
        return {'sat': 'error', 'sol': [], 'mul_sol': f'Parse Error: {e}', 'exe_time': '0ms'}
    
    if progress_callback:
        progress_callback({'status': 'Analyzing...', 'time': 0})
    
    # ============================================================================
    # NO PRUNING STRATEGY - Keep ALL users
    # ============================================================================
    # CRITICAL INSIGHT: The solution uses high-ID users (like u421, u372),
    # so we CANNOT prune by taking first N users.
    # Instead, rely on integer domain variables to keep model manageable.
    
    # Just use all users
    active_users = list(range(instance.num_users))
    
    if progress_callback:
        progress_callback({
            'status': 'Building model (all users)...',
            'time': time.time() - start_time
        })
    
    # Build validity map
    valid_for_step = {}
    for s in range(instance.num_steps):
        valid = [u for u in active_users if u not in instance.authorizations or s in instance.authorizations[u]]
        valid_for_step[s] = valid
        
        # Early UNSAT check
        if not valid:
            end_time = time.time()
            return {
                'sat': 'unsat',
                'sol': [],
                'mul_sol': f'Step {s+1} has no valid users',
                'exe_time': f"{int((end_time - start_time) * 1000)}ms"
            }
    
    # BoD conflict check
    for s1, s2 in instance.binding_duty:
        if not (set(valid_for_step[s1]) & set(valid_for_step[s2])):
            end_time = time.time()
            return {
                'sat': 'unsat',
                'sol': [],
                'mul_sol': f'BoD conflict between s{s1+1} and s{s2+1}',
                'exe_time': f"{int((end_time - start_time) * 1000)}ms"
            }
    
    if progress_callback:
        progress_callback({'status': 'Building model...', 'time': time.time() - start_time})
    
    # ============================================================================
    # MODEL WITH INTEGER VARIABLES
    # ============================================================================
    
    model = cp_model.CpModel()
    
    # Integer variables: p[s] = assigned user for step s
    p = {}
    for s in range(instance.num_steps):
        p[s] = model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(valid_for_step[s]),
            f'p{s}'
        )
    
    # Helper for boolean vars (created on demand)
    bool_cache = {}
    def get_bool(s, u):
        if (s, u) not in bool_cache:
            b = model.NewBoolVar(f'b{s}_{u}')
            model.Add(p[s] == u).OnlyEnforceIf(b)
            model.Add(p[s] != u).OnlyEnforceIf(b.Not())
            bool_cache[(s, u)] = b
        return bool_cache[(s, u)]
    
    # Constraints
    
    # 1. SoD: p[s1] != p[s2]
    for s1, s2 in instance.separation_duty:
        model.Add(p[s1] != p[s2])
    
    # 2. BoD: p[s1] == p[s2]
    for s1, s2 in instance.binding_duty:
        model.Add(p[s1] == p[s2])
    
    # 3. At-most-k
    for i, (k, steps) in enumerate(instance.at_most_k):
        if k >= len(steps):
            continue  # Trivially satisfied
        
        # Find relevant users
        relevant = set()
        for s in steps:
            relevant.update(valid_for_step[s])
        
        # Create user-used variables
        user_used = []
        for u in relevant:
            lits = [get_bool(s, u) for s in steps if u in valid_for_step[s]]
            if lits:
                u_var = model.NewBoolVar(f'amk{i}_u{u}')
                model.AddMaxEquality(u_var, lits)
                user_used.append(u_var)
        
        if user_used:
            model.Add(sum(user_used) <= k)
    
    # 4. One-team
    for i, (steps, teams) in enumerate(instance.one_team):
        viable = []
        for t_idx, team in enumerate(teams):
            # Check viability
            is_viable = all(
                any(u in valid_for_step[s] for u in team if u < instance.num_users)
                for s in steps
            )
            
            if is_viable:
                t_var = model.NewBoolVar(f'ot{i}_t{t_idx}')
                viable.append(t_var)
                
                for s in steps:
                    members = [get_bool(s, u) for u in team if u < instance.num_users and u in valid_for_step[s]]
                    if members:
                        model.Add(sum(members) == 1).OnlyEnforceIf(t_var)
        
        if not viable:
            end_time = time.time()
            return {
                'sat': 'unsat',
                'sol': [],
                'mul_sol': 'No viable team',
                'exe_time': f"{int((end_time - start_time) * 1000)}ms"
            }
        
        model.AddExactlyOne(viable)
    
    if progress_callback:
        progress_callback({
            'status': 'Solving...',
            'num_vars': len(model.Proto().variables),
            'num_constraints': len(model.Proto().constraints),
            'time': time.time() - start_time
        })
    
    # ============================================================================
    # SOLVER WITH AGGRESSIVE PARAMETERS
    # ============================================================================
    
    solver = cp_model.CpSolver()
    
    # Adaptive configuration based on problem size
    if instance.num_users >= 400:
        # HARD instances - use all optimizations
        solver.parameters.num_search_workers = 16  # Max parallelism
        solver.parameters.max_time_in_seconds = 180.0  # Give it 3 minutes
        solver.parameters.linearization_level = 0
        solver.parameters.cp_model_presolve = True
        solver.parameters.cp_model_probing_level = 2
        
        # Use SAT-specific parameters
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        solver.parameters.optimize_with_core = True
        solver.parameters.optimize_with_max_hs = True
        
        # Variable ordering: most constrained first
        step_order = sorted(range(instance.num_steps), key=lambda s: len(valid_for_step[s]))
        model.AddDecisionStrategy(
            [p[s] for s in step_order],
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MIN_VALUE
        )
    else:
        # Normal instances
        solver.parameters.num_search_workers = 8
        solver.parameters.max_time_in_seconds = 30.0
        solver.parameters.linearization_level = 0
    
    try:
        status = solver.Solve(model)
    except Exception as e:
        end_time = time.time()
        return {
            'sat': 'error',
            'sol': [],
            'mul_sol': f'Solver error: {e}',
            'exe_time': f"{int((end_time - start_time) * 1000)}ms"
        }
    
    # ============================================================================
    # EXTRACT RESULT
    # ============================================================================
    
    result = {'sat': 'unsat', 'sol': [], 'mul_sol': '', 'exe_time': ''}
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        result['sat'] = 'sat'
        sol_map = {}
        
        for s in range(instance.num_steps):
            u = solver.Value(p[s])
            result['sol'].append(f"s{s+1}: u{u+1}")
            sol_map[s] = u
        
        # Multiple solutions check (quick)
        blocking = [get_bool(s, u) for s, u in sol_map.items()]
        model.Add(sum(blocking) <= instance.num_steps - 1)
        
        solver.parameters.max_time_in_seconds = 0.5
        status2 = solver.Solve(model)
        
        if status2 in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            result['mul_sol'] = 'other solutions exist'
        else:
            result['mul_sol'] = 'this is the only solution'
    
    elif status == cp_model.INFEASIBLE:
        result['mul_sol'] = 'Problem is infeasible'
    elif status == cp_model.UNKNOWN:
        result['mul_sol'] = 'Timeout - problem too complex'
    
    end_time = time.time()
    result['exe_time'] = f"{int((end_time - start_time) * 1000)}ms"
    
    if progress_callback:
        progress_callback({'status': 'Finished', 'time': end_time - start_time})
    
    return result






def run_batch_analysis(directory="SAI/instances"):
    """
    Iterates through example1.txt to example19.txt in the given directory,
    runs the Solver, and prints a Markdown table of results.
    """
    print("| Instance | Status | Time (ms) |")
    print("| :--- | :--- | :--- |")
    
    for i in range(1, 20):
        filename = f"example{i}.txt"
        path = os.path.join(directory, filename)
        
        if os.path.exists(path):
            result = Solver(path)
            time_val = result['exe_time'].replace('ms', '')
            print(f"| {filename} | {result['sat']} | {time_val} |")
        else:
            print(f"| {filename} | Not Found | - |")

# ==========================================
# PART 3: GUI Application
# ==========================================

class TextRedirector(object):
    """Redirects text written to sys.stdout to a tkinter Text widget."""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state="disabled")
        # Force update to show progress in real-time
        self.widget.update_idletasks()

    def flush(self):
        pass

class WSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WSP Solver - SAI Coursework")
        self.root.geometry("600x500")
        
        # --- Top Frame: File Selection ---
        top_frame = tk.Frame(root, pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
        
        self.lbl_file = tk.Label(top_frame, text="No file selected", anchor="w", relief=tk.SUNKEN)
        self.lbl_file.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.btn_browse = tk.Button(top_frame, text="Browse", command=self.browse_file)
        self.btn_browse.pack(side=tk.LEFT)

        # --- Middle Frame: Actions ---
        action_frame = tk.Frame(root, pady=10)
        action_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
        
        self.btn_run = tk.Button(action_frame, text="Solve Instance", command=self.run_solver, bg="#dddddd", state=tk.DISABLED)
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.btn_batch = tk.Button(action_frame, text="Run Batch Analysis", command=self.run_batch, bg="#dddddd")
        self.btn_batch.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # --- Progress Frame ---
        progress_frame = tk.Frame(root, pady=5)
        progress_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
        
        self.lbl_progress_status = tk.Label(progress_frame, text="Ready", anchor="w", font=("Arial", 9))
        self.lbl_progress_status.pack(side=tk.TOP, fill=tk.X)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate', length=300)
        self.progress_bar.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        
        self.lbl_progress_info = tk.Label(progress_frame, text="", anchor="w", font=("Arial", 8), fg="gray")
        self.lbl_progress_info.pack(side=tk.TOP, fill=tk.X)

        # --- Bottom Frame: Output ---
        self.txt_output = scrolledtext.ScrolledText(root, state='disabled', font=("Consolas", 10))
        self.txt_output.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.selected_file_path = None
        self.solving = False

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            self.selected_file_path = file_path
            self.lbl_file.config(text=os.path.basename(file_path))
            self.btn_run.config(state=tk.NORMAL)

    def run_solver(self):
        if not self.selected_file_path or self.solving:
            return
        
        self.solving = True
        self.btn_run.config(state=tk.DISABLED)
        self.btn_batch.config(state=tk.DISABLED)
        
        # Reset progress
        self.lbl_progress_status.config(text="Initializing...", fg="black")
        self.lbl_progress_info.config(text="")
        self.progress_bar.stop()
        self.progress_bar['mode'] = 'indeterminate'
        self.progress_bar['value'] = 0
        self.progress_bar.start(10)
        
        self.txt_output.configure(state='normal')
        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, f"Solving {os.path.basename(self.selected_file_path)}...\n")
        self.txt_output.configure(state='disabled')
        
        # Threading to prevent GUI freeze
        t = threading.Thread(target=self._solve_thread)
        t.daemon = True
        t.start()
    
    def _update_progress(self, info):
        """Thread-safe progress update"""
        def update():
            status = info.get('status', 'Running...')
            
            # Update status label with color coding
            if 'Error' in status or 'error' in status.lower():
                self.lbl_progress_status.config(text=status, fg="red")
            elif 'Finished' in status:
                self.lbl_progress_status.config(text=status, fg="green")
            elif 'Found solution' in status:
                self.lbl_progress_status.config(text=status, fg="blue")
            else:
                self.lbl_progress_status.config(text=status, fg="black")
            
            # Build info string
            info_parts = []
            if 'num_vars' in info:
                info_parts.append(f"Variables: {info['num_vars']}")
            if 'num_constraints' in info:
                info_parts.append(f"Constraints: {info['num_constraints']}")
            if 'solutions' in info and info['solutions'] > 0:
                info_parts.append(f"Solutions: {info['solutions']}")
            if 'time' in info:
                elapsed = info['time']
                if elapsed > 0:
                    info_parts.append(f"Time: {elapsed:.2f}s")
            
            if info_parts:
                self.lbl_progress_info.config(text=" | ".join(info_parts))
            
            # Update progress bar
            if status == 'Finished':
                self.progress_bar.stop()
                self.progress_bar['mode'] = 'determinate'
                self.progress_bar['value'] = 100
            elif status == 'Starting solver...':
                self.progress_bar['mode'] = 'indeterminate'
                self.progress_bar.start(10)
            elif 'Found solution' in status:
                # Keep indeterminate mode while searching for more solutions
                if self.progress_bar['mode'] != 'indeterminate':
                    self.progress_bar['mode'] = 'indeterminate'
                    self.progress_bar.start(10)
        
        self.root.after(0, update)
        
    def _solve_thread(self):
        try:
            result = Solver(self.selected_file_path, progress_callback=self._update_progress)
            
            # Update progress to finished
            self._update_progress({'status': 'Finished', 'time': float(result['exe_time'].replace('ms', '')) / 1000})
            
            output_text = "\n--- Results ---\n"
            output_text += f"Status: {result['sat']}\n"
            output_text += f"Time:   {result['exe_time']}\n"
            if result['sat'] == 'sat':
                output_text += "Assignments:\n"
                for line in result['sol']:
                    output_text += f"  {line}\n"
                if result.get('mul_sol'):
                    output_text += f"\n{result['mul_sol']}\n"
            elif result['sat'] == 'error':
                output_text += f"Error: {result.get('mul_sol', 'Unknown error')}\n"
            else:
                output_text += "No solution found.\n"
                if result.get('mul_sol'):
                    output_text += f"{result['mul_sol']}\n"
                 
            self._safe_print(output_text)
            
        except Exception as e:
            self._update_progress({'status': 'Error', 'time': 0})
            self._safe_print(f"Error: {str(e)}\n")
        finally:
            self.solving = False
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_batch.config(state=tk.NORMAL))

    def run_batch(self):
        self.txt_output.configure(state='normal')
        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, "Running Batch Analysis...\n")
        self.txt_output.configure(state='disabled')
        
        t = threading.Thread(target=self._batch_thread)
        t.start()

    def _batch_thread(self):
        # Redirect stdout to text widget
        old_stdout = sys.stdout
        sys.stdout = TextRedirector(self.txt_output)
        
        try:
            # Check if default dir exists, if not try asking or use current
            if os.path.exists("SAI/instances"):
                run_batch_analysis("SAI/instances")
            else:
                # Fallback to current directory or warn
                print("Default directory 'SAI/instances' not found. Scanning current directory...")
                run_batch_analysis(".")
        except Exception as e:
            print(f"Batch Error: {e}")
        finally:
            sys.stdout = old_stdout
            
    def _safe_print(self, text):
        self.root.after(0, lambda: self._append_text(text))
        
    def _append_text(self, text):
        self.txt_output.configure(state='normal')
        self.txt_output.insert(tk.END, text)
        self.txt_output.see(tk.END)
        self.txt_output.configure(state='disabled')

if __name__ == "__main__":
    # If run as CLI with 'batch', do batch
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        run_batch_analysis()
    else:
        root = tk.Tk()
        app = WSPApp(root)
        root.mainloop()

