
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
    Optimized WSP Solver for wsp_app.py with Super User Pruning.
    FIXED: One-team user extraction logic.
    """
    start_time = time.time()
    
    # --- 1. Parse Instance ---
    try:
        # read_file is defined in this same file, so use it directly
        instance = read_file(filename)
    except Exception as e:
        return {'sat': 'error', 'sol': [], 'mul_sol': str(e), 'exe_time': '0ms'}
    
    model = cp_model.CpModel()
    
    # --- 2. SUPER USER PRUNING (Optimization) ---
    # Goal: Identify users who are mathematically identical "Super Users" and remove excess ones.
    
    # A. Find "Essential Users" (Those explicitly mentioned in constraints)
    essential_users = set(instance.authorizations.keys())
    
    # FIXED LOGIC HERE: Correctly iterate through teams
    for steps, teams in instance.one_team:
        for team_users in teams:     # Iterate through each team (which is a list of users)
            for u in team_users:     # Iterate through users in that team
                if u < instance.num_users: 
                    essential_users.add(u)
    
    # B. Find "Generic Users" (Implicit Super Users not in constraints)
    all_users_set = set(range(instance.num_users))
    generic_users = list(all_users_set - essential_users)
    generic_users.sort()

    # C. Prune: We only need enough super users to potentially cover all steps.
    # If there are 60 steps, 60 super users are sufficient. 
    # Keeping 400+ super users is what killed the performance before.
    needed_generic_count = min(len(generic_users), instance.num_steps)
    kept_generic_users = generic_users[:needed_generic_count]
    
    # D. Define the "Active Universe" for the solver
    active_users = list(essential_users) + kept_generic_users
    # active_users_set = set(active_users) # Not strictly needed but good for reference
    
    # --- 3. BUILD SPARSE MAPS ---
    # valid_users_for_step[s] will ONLY contain users from our pruned 'active_users' list
    valid_users_for_step = {}
    for s in range(instance.num_steps):
        valid = []
        for u in active_users:
            # Check Authorization:
            # 1. If user is in auth dict, step MUST be in their list.
            # 2. If user is NOT in auth dict (Super User), they can do anything.
            if u in instance.authorizations:
                if s in instance.authorizations[u]:
                    valid.append(u)
            else:
                valid.append(u) # Implicit Super User permission
        valid_users_for_step[s] = valid
    
    # --- 4. CREATE VARIABLES (Sparse) ---
    x = {} # x[s, u] -> boolean variable
    for s in range(instance.num_steps):
        for u in valid_users_for_step[s]:
            x[s, u] = model.NewBoolVar(f'x_s{s}_u{u}')
    
    # --- 5. CONSTRAINTS ---
    # (A) Assignment: Each step has exactly one user
    for s in range(instance.num_steps):
        step_vars = [x[s, u] for u in valid_users_for_step[s]]
        if not step_vars:
            # No valid users for this step -> UNSAT
            result = {'sat': 'unsat', 'sol': [], 'mul_sol': f'No valid users for step {s+1}', 'exe_time': ''}
            end_time = time.time()
            result['exe_time'] = f"{int((end_time - start_time) * 1000)}ms"
            return result
        model.AddExactlyOne(step_vars)
    
    # (B) SoD: x[s1, u] + x[s2, u] <= 1
    for s1, s2 in instance.separation_duty:
        # Only iterate intersection of valid users (Fast!)
        common = set(valid_users_for_step[s1]) & set(valid_users_for_step[s2])
        for u in common:
            if (s1, u) in x and (s2, u) in x:
                model.Add(x[s1, u] + x[s2, u] <= 1)
    
    # (C) BoD: x[s1, u] == x[s2, u]
    for s1, s2 in instance.binding_duty:
        u1_set = set(valid_users_for_step[s1])
        u2_set = set(valid_users_for_step[s2])
        
        # Users in both sets must match status
        for u in u1_set & u2_set:
            if (s1, u) in x and (s2, u) in x:
                model.Add(x[s1, u] == x[s2, u])
                
        # Users unique to one set must be 0 (cannot be assigned)
        for u in u1_set - u2_set:
             if (s1, u) in x: model.Add(x[s1, u] == 0)
        for u in u2_set - u1_set:
             if (s2, u) in x: model.Add(x[s2, u] == 0)
    
    # (D) At-most-k (Optimized Loop)
    for i, (k, steps) in enumerate(instance.at_most_k):
        # 1. Identify users relevant to THESE steps only
        involved_users = set()
        for s in steps:
            involved_users.update(valid_users_for_step[s])
        
        # 2. Only create auxiliary variables for these relevant users
        used_vars = []
        for u in involved_users:
            # Collect x[s,u] for all steps in this constraint
            step_vars = [x[s, u] for s in steps if (s, u) in x]
            
            if step_vars:
                u_active = model.NewBoolVar(f'amk_{i}_u{u}')
                model.AddMaxEquality(u_active, step_vars)
                used_vars.append(u_active)
        
        if used_vars:
            model.Add(sum(used_vars) <= k)
    
    # (E) One-team (Optimized Loop with pre-computed capabilities)
    for i, (steps, teams) in enumerate(instance.one_team):
        team_vars = []
        team_capabilities = {}  # Pre-compute which teams can do which steps
        
        for t_idx, team_users in enumerate(teams):
            # Optimization: Check if team is viable FIRST
            # A team is viable if for every step, there is at least one member who can do it.
            is_viable = True
            team_capabilities[t_idx] = {}
            
            for s in steps:
                capable_users_vars = []
                for u in team_users:
                    if u < instance.num_users and (s, u) in x:
                        capable_users_vars.append(x[s, u])
                
                if capable_users_vars:
                    team_capabilities[t_idx][s] = capable_users_vars
                else:
                    is_viable = False
                    break
            
            if is_viable:
                t_var = model.NewBoolVar(f'ot_{i}_t{t_idx}')
                team_vars.append(t_var)
                
                # If this team is selected (t_var=1), then for each step, 
                # exactly one user FROM THIS TEAM must be assigned.
                for s in steps:
                    capable_users_vars = team_capabilities[t_idx][s]
                    # Use OnlyEnforceIf for better propagation
                    model.Add(sum(capable_users_vars) == 1).OnlyEnforceIf(t_var)
            
        if team_vars:
            model.AddExactlyOne(team_vars)
        else:
            # No team can satisfy the requirements -> UNSAT
            result = {'sat': 'unsat', 'sol': [], 'mul_sol': 'no viable teams', 'exe_time': ''}
            end_time = time.time()
            result['exe_time'] = f"{int((end_time - start_time) * 1000)}ms"
            return result
    
    # --- 6. SOLVE ---
    solver = cp_model.CpSolver()
    
    # Create callback if progress monitoring is requested
    callback = None
    if progress_callback:
        callback = SolutionCallback(progress_callback)
    
    # Optimize solver parameters for hard instances
    solver.parameters.num_search_workers = 8  # Parallel search
    
    # Set timeout for hard instances (5 minutes)
    if instance.num_steps >= 50 or instance.num_constraints >= 500:
        solver.parameters.max_time_in_seconds = 300.0
        # Use portfolio search for hard instances (tries multiple strategies)
        solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
        # Better linearization for complex constraints
        solver.parameters.linearization_level = 2
    else:
        # Shorter timeout for easier instances
        solver.parameters.max_time_in_seconds = 30.0
    
    # Update progress callback with solver stats
    if callback and progress_callback:
        progress_callback({
            'status': 'Starting solver...',
            'num_vars': len(model.Proto().variables),
            'num_constraints': len(model.Proto().constraints),
            'time': 0
        })
    
    # Start periodic progress updates in a separate thread
    progress_stop_event = threading.Event()
    if callback and progress_callback:
        def periodic_update():
            while not progress_stop_event.is_set():
                time.sleep(0.5)  # Update every 500ms
                if not progress_stop_event.is_set():
                    stats = callback.get_stats()
                    stats['num_vars'] = len(model.Proto().variables)
                    stats['num_constraints'] = len(model.Proto().constraints)
                    progress_callback(stats)
        
        progress_thread = threading.Thread(target=periodic_update, daemon=True)
        progress_thread.start()
    
    # solver.parameters.log_search_progress = True # Uncomment for debugging
    
    # Solve with callback
    try:
        if callback:
            status = solver.Solve(model, callback)
        else:
            status = solver.Solve(model)
    finally:
        # Stop periodic updates
        if callback:
            progress_stop_event.set()
    
    # Final progress update
    if callback and progress_callback:
        elapsed = time.time() - callback.start_time
        progress_callback({
            'status': 'Finished',
            'time': elapsed,
            'solutions': callback.solution_count
        })
    
    # --- 7. OUTPUT ---
    result = {'sat': 'unsat', 'sol': [], 'mul_sol': '', 'exe_time': ''}
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        result['sat'] = 'sat'
        current_assignment = []
        
        # Reconstruct solution map
        for s in range(instance.num_steps):
            for u in valid_users_for_step[s]:
                if solver.Value(x[s, u]):
                    # Convert 0-index back to 1-index strings
                    result['sol'].append(f"s{s+1}: u{u+1}")
                    current_assignment.append(x[s, u])
                    break
        
        # Check for multiple solutions (blocking clause) - skip for hard instances to save time
        if instance.num_steps < 50 and instance.num_constraints < 500:
            # Force at least one assignment to change
            if current_assignment:
                model.Add(sum(current_assignment) < len(current_assignment))
                status2 = solver.Solve(model)
                if status2 == cp_model.OPTIMAL or status2 == cp_model.FEASIBLE:
                    result['mul_sol'] = 'other solutions exist'
                else:
                    result['mul_sol'] = 'this is the only solution'
        else:
            result['mul_sol'] = 'solution found (multiple solution check skipped for performance)'
    elif status == cp_model.MODEL_INVALID:
        result['sat'] = 'error'
        result['mul_sol'] = 'Model is invalid'
    elif status == cp_model.INFEASIBLE:
        result['sat'] = 'unsat'
        result['mul_sol'] = 'Problem is infeasible'
    elif status == cp_model.UNKNOWN:
        result['sat'] = 'unknown'
        result['mul_sol'] = 'Solver could not determine satisfiability (may have timed out)'
    
    end_time = time.time()
    result['exe_time'] = f"{int((end_time - start_time) * 1000)}ms"
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

