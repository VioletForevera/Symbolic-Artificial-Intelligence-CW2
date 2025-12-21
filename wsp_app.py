
import re
import time
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext
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

def Solver(filename):
    start_time = time.time()
    try:
        instance = read_file(filename)
    except Exception as e:
        return {'sat': 'error', 'sol': [], 'mul_sol': '', 'exe_time': '0ms'}
    
    model = cp_model.CpModel()
    
    # --- 1. SUPER USER PRUNING (The Fix for Hard Instances) ---
    # Identify users who MUST be there (mentioned in constraints)
    essential_users = set(instance.authorizations.keys())
    for steps, team_users in instance.one_team:
        for u in team_users:
            if u < instance.num_users: # Safety check
                essential_users.add(u)
    
    # Identify "Super Users" (mathematically identical placeholders)
    all_users_set = set(range(instance.num_users))
    generic_users = list(all_users_set - essential_users)
    generic_users.sort()
    
    # We only need enough super users to cover the steps. 
    # If there are 60 steps, 60 super users are enough. 300 is wasteful.
    needed_generic = generic_users[:instance.num_steps]
    active_users = list(essential_users) + needed_generic
    active_users_set = set(active_users)
    
    # --- 2. BUILD SPARSE MAPS ---
    # Only map valid users. If a user is not in 'active_users', they don't exist in the model.
    valid_users_for_step = {}
    for s in range(instance.num_steps):
        valid = []
        for u in active_users:
            # Logic: If u is in authorizations, check if s is allowed.
            # If u is NOT in authorizations (Super User), they are allowed everything.
            if u in instance.authorizations:
                if s in instance.authorizations[u]:
                    valid.append(u)
            else:
                valid.append(u) # Super user
        valid_users_for_step[s] = valid
    
    # --- 3. CREATE VARIABLES ---
    x = {} # x[s, u]
    for s in range(instance.num_steps):
        for u in valid_users_for_step[s]:
            x[s, u] = model.NewBoolVar(f'x_{s}_{u}')
    
    # --- 4. CONSTRAINTS ---
    # (A) Assignment: Each step has exactly one user
    for s in range(instance.num_steps):
        model.AddExactlyOne([x[s, u] for u in valid_users_for_step[s]])
    
    # (B) SoD: x[s1, u] + x[s2, u] <= 1
    for s1, s2 in instance.separation_duty:
        common = set(valid_users_for_step[s1]) & set(valid_users_for_step[s2])
        for u in common:
            model.Add(x[s1, u] + x[s2, u] <= 1)
    
    # (C) BoD: x[s1, u] == x[s2, u]
    for s1, s2 in instance.binding_duty:
        u1 = set(valid_users_for_step[s1])
        u2 = set(valid_users_for_step[s2])
        for u in u1 | u2:
            if u in u1 and u in u2:
                model.Add(x[s1, u] == x[s2, u])
            elif u in u1:
                model.Add(x[s1, u] == 0)
            elif u in u2:
                model.Add(x[s2, u] == 0)
    
    # (D) At-most-k (Optimized Loop)
    for i, (k, steps) in enumerate(instance.at_most_k):
        # Only create vars for users relevant to these steps
        relevant_users = set()
        for s in steps:
            relevant_users.update(valid_users_for_step[s])
        
        used_vars = []
        for u in relevant_users:
            # Does user u do ANY of the steps?
            step_vars = [x[s, u] for s in steps if (s, u) in x]
            if step_vars:
                u_active = model.NewBoolVar(f'amk_{i}_u{u}')
                model.AddMaxEquality(u_active, step_vars)
                used_vars.append(u_active)
        model.Add(sum(used_vars) <= k)
    
    # (E) One-team
    for i, (steps, teams) in enumerate(instance.one_team):
        team_vars = []
        for t_idx, team_users in enumerate(teams):
            # Optimization: Check if team is viable
            is_viable = True
            for s in steps:
                # Can ANYone in the team do step s?
                can_do = False
                for u in team_users:
                    if u < instance.num_users and u in valid_users_for_step[s]:
                        can_do = True
                        break
                if not can_do:
                    is_viable = False
                    break
            
            if is_viable:
                t_var = model.NewBoolVar(f'ot_{i}_t{t_idx}')
                team_vars.append(t_var)
                # Enforce team logic
                for s in steps:
                    # Users in this team who can do step s
                    capable = [x[s, u] for u in team_users if u < instance.num_users and (s, u) in x]
                    model.Add(sum(capable) == 1).OnlyEnforceIf(t_var)
            # If not viable, we just don't create t_var, effectively 0
            
        if team_vars:
            model.AddExactlyOne(team_vars)
        else:
            model.Add(0 == 1) # Impossible instance
    
    # --- 5. SOLVE ---
    solver = cp_model.CpSolver()
    # Enable multithreading
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    
    # --- 6. OUTPUT ---
    result = {'sat': 'unsat', 'sol': [], 'mul_sol': '', 'exe_time': ''}
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        result['sat'] = 'sat'
        current_sol_vars = []
        for s in range(instance.num_steps):
            for u in valid_users_for_step[s]:
                if solver.Value(x[s, u]):
                    result['sol'].append(f"s{s+1}: u{u+1}")
                    current_sol_vars.append(x[s, u])
                    break
        
        # Double check for multiple solutions
        model.Add(sum(current_sol_vars) < instance.num_steps)
        status2 = solver.Solve(model)
        if status2 == cp_model.OPTIMAL or status2 == cp_model.FEASIBLE:
            result['mul_sol'] = 'other solutions exist'
        else:
            result['mul_sol'] = 'this is the only solution'
    
    end_time = time.time()
    result['exe_time'] = str(int((end_time - start_time) * 1000)) + 'ms'
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

        # --- Bottom Frame: Output ---
        self.txt_output = scrolledtext.ScrolledText(root, state='disabled', font=("Consolas", 10))
        self.txt_output.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.selected_file_path = None

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            self.selected_file_path = file_path
            self.lbl_file.config(text=os.path.basename(file_path))
            self.btn_run.config(state=tk.NORMAL)

    def run_solver(self):
        if not self.selected_file_path:
            return
            
        self.txt_output.configure(state='normal')
        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, f"Solving {os.path.basename(self.selected_file_path)}...\n")
        self.txt_output.configure(state='disabled')
        
        # Threading to prevent GUI freeze
        t = threading.Thread(target=self._solve_thread)
        t.start()
        
    def _solve_thread(self):
        try:
            result = Solver(self.selected_file_path)
            
            output_text = "\n--- Results ---\n"
            output_text += f"Status: {result['sat']}\n"
            output_text += f"Time:   {result['exe_time']}\n"
            if result['sat'] == 'sat':
                output_text += "Assignments:\n"
                for line in result['sol']:
                    output_text += f"  {line}\n"
            else:
                 output_text += "No solution found.\n"
                 
            self._safe_print(output_text)
            
        except Exception as e:
            self._safe_print(f"Error: {str(e)}")

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

