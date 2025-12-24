import re
import time
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from collections import defaultdict
# 使用标准库类型提示，避免 import * 造成的冲突
from typing import Optional, Any, Callable, Union

# OR-Tools
from ortools.sat.python import cp_model

# Z3 (Explicit import to avoid namespace pollution)
import z3

# Optional Visualization


# ==========================================
# PART 1: Parser Logic
# ==========================================

class Instance:
    def __init__(self, num_steps: int, num_users: int, num_constraints: int):
        self.num_steps: int = num_steps
        self.num_users: int = num_users
        self.num_constraints: int = num_constraints
        
        # User ID -> list of authorized Step IDs
        self.authorizations: dict[int, list[int]] = {} 
        
        # List of (step1, step2) tuples
        self.separation_duty: list[tuple[int, int]] = []
        
        # List of (step1, step2) tuples
        self.binding_duty: list[tuple[int, int]] = []
        
        # List of (k, [step_ids]) tuples
        self.at_most_k: list[tuple[int, list[int]]] = []
        
        # List of (step_ids, [team1_users, team2_users, ...]) tuples
        self.one_team: list[tuple[list[int], list[list[int]]]] = []

    def __repr__(self) -> str:
        return (f"Instance(Steps={self.num_steps}, Users={self.num_users}, "
                f"Constraints={self.num_constraints}, "
                f"Auths={len(self.authorizations)}, "
                f"SoD={len(self.separation_duty)}, "
                f"BoD={len(self.binding_duty)}, "
                f"AMK={len(self.at_most_k)}, "
                f"OneTeam={len(self.one_team)})")

def read_file(filename: str) -> Optional[Instance]:
    """
    Ultra-robust parser that handles loose formatting and missing prefixes.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    instance = None
    
    # 1. First pass: Try to find header information
    n_steps = None
    n_users = None
    n_const = None

    # Regex to extract numbers from header lines like "# Steps: 10"
    re_steps = re.compile(r'Steps.*?(\d+)', re.IGNORECASE)
    re_users = re.compile(r'Users.*?(\d+)', re.IGNORECASE)
    re_const = re.compile(r'Constraints.*?(\d+)', re.IGNORECASE)

    for line in lines:
        if line.strip().startswith('%'): continue
        
        if n_steps is None:
            m = re_steps.search(line)
            if m: n_steps = int(m.group(1))
        if n_users is None:
            m = re_users.search(line)
            if m: n_users = int(m.group(1))
        if n_const is None:
            m = re_const.search(line)
            if m: n_const = int(m.group(1))
            
        if n_steps and n_users and n_const:
            instance = Instance(n_steps, n_users, n_const)
            break
    
    if instance is None:
        return None

    # 2. Second pass: Parse constraints
    # Helper to extract ALL integers from a string
    def get_ints(s):
        return [int(x) for x in re.findall(r'\d+', s)]

    for line in lines:
        line = line.strip()
        if not line or line.startswith('%') or line.startswith('#'):
            continue
            
        # Normalize line to lowercase for checking keywords
        lower_line = line.lower()
        
        # 1. Authorizations
        # Format: "authorizations u1 s1 s2" OR "authorizations 1 1 2"
        if lower_line.startswith('authori'):
            nums = get_ints(line)
            if len(nums) >= 1:
                # First number is User ID (1-based), rest are Step IDs (1-based)
                u_id = nums[0] - 1
                steps = [x - 1 for x in nums[1:]]
                
                if u_id not in instance.authorizations:
                    instance.authorizations[u_id] = []
                instance.authorizations[u_id].extend(steps)
            continue
            
        # 2. Separation of Duty
        # Format: "separation-of-duty s1 s2"
        if lower_line.startswith('separation'):
            nums = get_ints(line)
            if len(nums) >= 2:
                instance.separation_duty.append((nums[0]-1, nums[1]-1))
            continue
            
        # 3. Binding of Duty
        if lower_line.startswith('binding'):
            nums = get_ints(line)
            if len(nums) >= 2:
                instance.binding_duty.append((nums[0]-1, nums[1]-1))
            continue
            
        # 4. At-most-k
        # Format: "at-most-k 2 s1 s2 s3"
        if lower_line.startswith('at-most'):
            nums = get_ints(line)
            if len(nums) >= 2:
                k = nums[0]
                steps = [x - 1 for x in nums[1:]]
                instance.at_most_k.append((k, steps))
            continue
            
        # 5. One-team
        # Format: "one-team s1 s2 (u1 u2) (u3 u4)"
        # This is complex because of parentheses grouping teams
        if lower_line.startswith('one-team'):
            # Remove "one-team" prefix
            content = line[8:].strip()
            
            # Split by '(' to separate steps from teams
            parts = content.split('(')
            
            # First part contains steps
            steps_part = parts[0]
            steps = [x - 1 for x in get_ints(steps_part)]
            
            teams = []
            # Remaining parts are teams "(u1 u2)..."
            for team_part in parts[1:]:
                # Extract numbers from inside the parens
                team_users = [x - 1 for x in get_ints(team_part)]
                if team_users:
                    teams.append(team_users)
            
            if steps and teams:
                instance.one_team.append((steps, teams))
            continue

    # Debug print removed
    return instance

# ==========================================
# PART 2: Specialized Solvers
# ==========================================

def Solver_Z3_Bool(filename: str, progress_callback: Optional[Callable] = None, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Alternative Formulation: Z3 Pure Boolean (SAT) Solver.
    Uses pure Boolean variables x[s][u] and Pseudo-Boolean constraints.
    """
    start_time = time.time()
    
    try:
        instance = read_file(filename)
    except Exception as e:
        return {'sat': 'error', 'sol': [], 'mul_sol': f'Parse error: {e}', 'exe_time': '0ms'}
    
    if progress_callback:
        progress_callback({'status': 'Building Z3 SAT model...', 'time': 0})
    
    # 1. Pre-computation of Valid Users
    valid_users_for_step = {}
    for s in range(instance.num_steps):
        valid = []
        for u in range(instance.num_users):
            if u in instance.authorizations:
                if s in instance.authorizations[u]:
                    valid.append(u)
            else:
                valid.append(u)
        valid_users_for_step[s] = valid
        
        if not valid:
            return {'sat': 'unsat', 'sol': [], 'mul_sol': f'Step {s+1} has no valid users', 'exe_time': '0ms'}

    solver = z3.Solver()
    
    # Timeout

    
    # 2. Variable Creation: x[s][u]
    x = {} 
    
    for s in range(instance.num_steps):
        for u in valid_users_for_step[s]:
            x[(s, u)] = z3.Bool(f'x_{s}_{u}')
    
    def get_x(s, u):
        return x.get((s, u), z3.BoolVal(False))

    # 3. Constraints

    # Constraint 0: Exactly one user per step
    for s in range(instance.num_steps):
        vars_s = [x[(s, u)] for u in valid_users_for_step[s]]
        solver.add(z3.PbEq([(v, 1) for v in vars_s], 1))

    # Constraint 1: SoD
    for s1, s2 in instance.separation_duty:
        common_users = set(valid_users_for_step[s1]) & set(valid_users_for_step[s2])
        for u in common_users:
            solver.add(z3.Not(z3.And(x[(s1, u)], x[(s2, u)])))

    # Constraint 2: BoD
    for s1, s2 in instance.binding_duty:
        u_s1 = set(valid_users_for_step[s1])
        u_s2 = set(valid_users_for_step[s2])
        all_u = u_s1 | u_s2
        for u in all_u:
            v1 = get_x(s1, u)
            v2 = get_x(s2, u)
            solver.add(v1 == v2)

    # Constraint 3: At-Most-K
    for i, (k, steps) in enumerate(instance.at_most_k):
        if k >= len(steps): continue

        involved_users = set()
        for s in steps:
            involved_users.update(valid_users_for_step[s])
        
        u_used_vars = []
        for u in involved_users:
            u_active = z3.Bool(f'amk_{i}_u{u}')
            steps_u_can_do = [s for s in steps if u in valid_users_for_step[s]]
            if not steps_u_can_do:
                solver.add(z3.Not(u_active))
            else:
                assignments = [x[(s, u)] for s in steps_u_can_do]
                solver.add(u_active == z3.Or(assignments))
                u_used_vars.append(u_active)
        
        if u_used_vars:
            solver.add(z3.PbLe([(v, 1) for v in u_used_vars], k))

    # Constraint 4: One-Team
    for i, (steps, teams) in enumerate(instance.one_team):
        viable_team_vars = []
        for t_idx, team_users in enumerate(teams):
            is_viable = all(
                any(u in valid_users_for_step[s] for u in team_users if u < instance.num_users)
                for s in steps
            )
            
            if is_viable:
                t_var = z3.Bool(f'ot_{i}_{t_idx}')
                viable_team_vars.append(t_var)
                
                for s in steps:
                    valid_team_members = [u for u in team_users if u in valid_users_for_step[s]]
                    if valid_team_members:
                        solver.add(z3.Implies(t_var, z3.Or([x[(s, u)] for u in valid_team_members])))
                    else:
                        solver.add(z3.Not(t_var)) 

        if not viable_team_vars:
             return {'sat': 'unsat', 'sol': [], 'mul_sol': 'No viable team', 'exe_time': '0ms'}

        solver.add(z3.PbEq([(v, 1) for v in viable_team_vars], 1))

    if progress_callback:
        progress_callback({
            'status': 'Solving with Z3 (Boolean)...',
            'assertions': len(solver.assertions()),
            'time': time.time() - start_time
        })
    
    # 4. Solving
    result = {'sat': 'unsat', 'sol': [], 'all_solutions': [], 'mul_sol': '', 'exe_time': ''}
    
    sol_limit = 5
    if config and 'solution_limit' in config:
        sol_limit = int(config['solution_limit'])

    found_solutions = []
    
    while len(found_solutions) < sol_limit:
        if solver.check() != z3.sat:
            break

        model = solver.model()
        
        current_sol = []
        blocking_clauses = []
        
        for s in range(instance.num_steps):
            assigned_u = -1
            # Find which user is assigned
            for u in valid_users_for_step[s]:
                if z3.is_true(model[x[(s, u)]]):
                    assigned_u = u
                    # Add to blocking: (NOT x_s_u)
                    # Because x_s_u is currently TRUE, we want to forbid this exact combination
                    blocking_clauses.append(z3.Not(x[(s, u)]))
                    break
            
            if assigned_u != -1:
                current_sol.append(f"s{s+1}: u{assigned_u+1}")
            else:
                current_sol.append(f"s{s+1}: u?")
        
        found_solutions.append(current_sol)
        
        # Add blocking clause to forbid THIS specific full assignment
        # Logic: It cannot be the case that ALL variables have these values again
        solver.add(z3.Or(blocking_clauses))
    
    if found_solutions:
        result['sat'] = 'sat'
        result['sol'] = found_solutions[0] # Best/first solution
        result['all_solutions'] = found_solutions
        result['mul_sol'] = f'Found {len(found_solutions)} solution(s).'
        
        # Check if more exist (if we hit the limit)
        if len(found_solutions) == sol_limit:
            if solver.check() == z3.sat:
                result['mul_sol'] += " (More exist)"
            else:
                result['mul_sol'] += " (No more solutions)"
        else:
            result['mul_sol'] += " (All found)"
    
    elif not found_solutions: 
        # Truly UNSAT
        pass # result['sat'] is already 'unsat'

    end_time = time.time()
    result['exe_time'] = f"{int((end_time - start_time) * 1000)}ms"
    
    if progress_callback:
        progress_callback({'status': 'Finished', 'time': end_time - start_time})
    
    return result

def Solver_SmartBacktrack(filename: str, progress_callback: Optional[Callable] = None, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Intelligent Backtracking Solver with Forward Checking & Heuristics.
    """
    start_time = time.time()
    
    try:
        instance = read_file(filename)
    except Exception as e:
        return {'sat': 'error', 'sol': [], 'mul_sol': f'Parse error: {e}', 'exe_time': '0ms'}
    
    if progress_callback:
        progress_callback({'status': 'Initializing backtrack search...', 'time': 0})
    
    # 1. Domain Filtering
    domains = {}
    for s in range(instance.num_steps):
        valid = []
        for u in range(instance.num_users):
            if u in instance.authorizations:
                if s in instance.authorizations[u]:
                    valid.append(u)
            else:
                valid.append(u)
        
        if not valid:
            return {'sat': 'unsat', 'sol': [], 'mul_sol': f'No valid users for step {s+1}', 'exe_time': '0ms'}
        
        domains[s] = set(valid)
    
    # BoD Propagation
    for s1, s2 in instance.binding_duty:
        common = domains[s1] & domains[s2]
        if not common:
            return {'sat': 'unsat', 'sol': [], 'mul_sol': 'BoD conflict', 'exe_time': '0ms'}
        domains[s1] = common
        domains[s2] = common
    
    # SoD Graph
    sod_neighbors = defaultdict(set)
    for s1, s2 in instance.separation_duty:
        sod_neighbors[s1].add(s2)
        sod_neighbors[s2].add(s1)

    # 2. Pre-process Global Constraints
    amk_map = defaultdict(list)
    for idx, (k, steps) in enumerate(instance.at_most_k):
        step_set = set(steps)
        for s in steps:
            amk_map[s].append((idx, k, step_set))

    ot_map = defaultdict(list)
    for idx, (steps, teams) in enumerate(instance.one_team):
        team_sets = [set(t) for t in teams]
        step_set = set(steps)
        for s in steps:
            ot_map[s].append((idx, step_set, team_sets))

    if progress_callback:
        progress_callback({'status': 'Starting search...', 'time': time.time() - start_time})
    
    # 3. Search
    assignment = {}
    nodes_explored = [0]
    sol_limit = 5
    if config:
        if 'solution_limit' in config: sol_limit = int(config['solution_limit'])

    found_solutions = []

    def backtrack(step_idx, steps_to_assign):
        nonlocal nodes_explored
        
        nodes_explored[0] += 1
        
        if progress_callback and nodes_explored[0] % 2000 == 0:
            progress_callback({
                'status': f'Explored {nodes_explored[0]} nodes ({len(found_solutions)} found)...',
                'time': time.time() - start_time,
                'solutions': len(found_solutions)
            })
        
        if step_idx >= len(steps_to_assign):
            # FOUND A SOLUTION
            current_sol = []
            for s in range(instance.num_steps):
                current_sol.append(f"s{s+1}: u{assignment[s]+1}")
            found_solutions.append(current_sol)
            
            if len(found_solutions) >= sol_limit:
                return True # Stop, limit reached
            return False # Continue searching
        
        s = steps_to_assign[step_idx]
        candidates = list(domains[s])
        
        # Heuristic: Sort candidates (Greedy AMK)
        used_users = set(assignment.values())
        candidates.sort(key=lambda u: (u not in used_users, u))
        
        for user in candidates:
            # SoD Check
            is_sod_violated = False
            for neighbor in sod_neighbors[s]:
                if neighbor in assignment and assignment[neighbor] == user:
                    is_sod_violated = True
                    break
            if is_sod_violated:
                continue

            # Incremental At-Most-K Check
            amk_violated = False
            for (amk_idx, k, involved_steps) in amk_map[s]:
                current_users = {user}
                for other_s in involved_steps:
                    if other_s in assignment:
                        current_users.add(assignment[other_s])
                
                if len(current_users) > k:
                    amk_violated = True
                    break
            if amk_violated:
                continue

            # Incremental One-Team Check
            ot_violated = False
            for (ot_idx, involved_steps, teams) in ot_map[s]:
                active_workers = {user}
                for other_s in involved_steps:
                    if other_s in assignment:
                        active_workers.add(assignment[other_s])
                
                curr_viable = False
                for team_set in teams:
                    if active_workers.issubset(team_set):
                        curr_viable = True
                        break
                
                if not curr_viable:
                    ot_violated = True
                    break
            if ot_violated:
                continue

            # Commit & Recurse
            assignment[s] = user
            
            if backtrack(step_idx + 1, steps_to_assign):
                return True # Stop signal propagated
            
            del assignment[s]
        
        return False
    
    # MRV Heuristic
    steps_to_assign = sorted(range(instance.num_steps),
                            key=lambda s: (len(domains[s]), -len(sod_neighbors[s])))
    
    backtrack(0, steps_to_assign)
    
    result = {'sat': 'unsat', 'sol': [], 'all_solutions': [], 'mul_sol': '', 'exe_time': ''}
    
    if found_solutions:
        result['sat'] = 'sat'
        result['sol'] = found_solutions[0]
        result['all_solutions'] = found_solutions
        
        result['mul_sol'] = f'Found {len(found_solutions)} solution(s) (explored {nodes_explored[0]} nodes)'
        if len(found_solutions) >= sol_limit:
            result['mul_sol'] += " (Limit reached)"
        else:
            result['mul_sol'] += " (All found)"
    else:
        result['mul_sol'] = f'No solution found (explored {nodes_explored[0]} nodes)'
    
    end_time = time.time()
    result['exe_time'] = f"{int((end_time - start_time) * 1000)}ms"
    
    if progress_callback:
        progress_callback({'status': 'Finished', 'time': end_time - start_time})
    
    return result

def Solver_Z3(filename: str, progress_callback: Optional[Callable] = None, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Z3 SMT Solver (Integer based)."""
    start_time = time.time()
    try:
        instance = read_file(filename)
    except Exception as e:
        return {'sat': 'error', 'sol': [], 'mul_sol': f'Parse error: {e}', 'exe_time': '0ms'}
    
    if not instance:
        return {'sat': 'error', 'sol': [], 'mul_sol': 'Parser returned None', 'exe_time': '0ms'}

    if progress_callback: progress_callback({'status': 'Building Z3 model...', 'time': 0})
    
    # Pre-calculate valid users per step
    valid_users_for_step = {}
    for s in range(instance.num_steps):
        valid = []
        for u in range(instance.num_users):
            # Implicit authorization rule:
            # If user is in dict, check list. If NOT in dict, they are superuser (allowed for all).
            if u in instance.authorizations:
                if s in instance.authorizations[u]: valid.append(u)
            else:
                valid.append(u)
        valid_users_for_step[s] = valid
        
        if not valid: 
            return {'sat': 'unsat', 'sol': [], 'mul_sol': f'Step {s+1} has no valid users (check authorizations)', 'exe_time': '0ms'}
    
    solver = z3.Solver()

    
    # Variables
    p = {}
    for s in range(instance.num_steps):
        p[s] = z3.Int(f'p_{s}')
        # Domain Constraint
        solver.add(z3.Or([p[s] == u for u in valid_users_for_step[s]]))
    
    # Constraints
    for s1, s2 in instance.separation_duty: 
        solver.add(p[s1] != p[s2])
        
    for s1, s2 in instance.binding_duty: 
        solver.add(p[s1] == p[s2])
    
    for i, (k, steps) in enumerate(instance.at_most_k):
        if time.time() - start_time > timeout_sec:
             return {'sat': 'unknown', 'sol': [], 'mul_sol': 'Timeout during model build (AMK)', 'exe_time': f"{int((time.time() - start_time)*1000)}ms"}
        if k >= len(steps): continue
        involved_users = set()
        for s in steps: involved_users.update(valid_users_for_step[s])
        
        # For each involved user, create a Bool indicating if they are used
        user_vars = []
        for u in involved_users:
            u_used = z3.Bool(f'amk_{i}_u{u}')
            # u_used is TRUE if p[s]==u for ANY relevant step
            conds = [p[s] == u for s in steps if u in valid_users_for_step[s]]
            if conds:
                solver.add(u_used == z3.Or(conds))
                user_vars.append(u_used)
        
        if user_vars:
            # PB Constraint: Sum of True booleans <= k
            solver.add(z3.PbLe([(v, 1) for v in user_vars], k))
    
    for i, (steps, teams) in enumerate(instance.one_team):
        viable_team_vars = []
        for t_idx, team_users in enumerate(teams):
            # Quick check if team is theoretically capable
            is_viable = all(any(u in valid_users_for_step[s] for u in team_users if u < instance.num_users) for s in steps)
            if is_viable:
                team_var = z3.Bool(f'team_{i}_{t_idx}')
                viable_team_vars.append(team_var)
                for s in steps:
                    team_members = [u for u in team_users if u < instance.num_users and u in valid_users_for_step[s]]
                    if team_members: 
                        # If team selected, step s MUST be done by a team member
                        solver.add(z3.Implies(team_var, z3.Or([p[s] == u for u in team_members])))
                    else:
                        # Impossible for this team to cover step s
                        solver.add(z3.Not(team_var))

        if not viable_team_vars: 
            return {'sat': 'unsat', 'sol': [], 'mul_sol': 'No viable team for constraint', 'exe_time': '0ms'}
        
        # Exactly one team selected (Sum == 1)
        solver.add(z3.PbEq([(v, 1) for v in viable_team_vars], 1))
    
    # Solve Loop
    result = {'sat': 'unsat', 'sol': [], 'all_solutions': [], 'mul_sol': '', 'exe_time': ''}
    sol_limit = int(config.get('solution_limit', 5)) if config else 5
    
    found_solutions = []
    
    if progress_callback: progress_callback({'status': 'Searching for solutions...', 'time': time.time() - start_time})

    while len(found_solutions) < sol_limit:
        if solver.check() == z3.sat:
            model = solver.model()
            current_sol = []
            blocking = []
            for s in range(instance.num_steps):
                user_val = model[p[s]].as_long()
                current_sol.append(f"s{s+1}: u{user_val+1}")
                blocking.append(p[s] != user_val)
            
            found_solutions.append(current_sol)
            solver.add(z3.Or(blocking)) # Block this exact assignment
        else:
            break
        
    if found_solutions:
        result['sat'] = 'sat'
        result['sol'] = found_solutions[0]
        result['all_solutions'] = found_solutions
        result['mul_sol'] = f'Found {len(found_solutions)} solution(s).'
        if len(found_solutions) == sol_limit:
             if solver.check() == z3.sat: result['mul_sol'] += " (More exist)"
             else: result['mul_sol'] += " (No more)"
        else:
             result['mul_sol'] += " (All found)"
    else:
        if str(solver.reason_unknown()) != 'no reason': result['mul_sol'] = f'Z3 Unknown: {solver.reason_unknown()}'
        else: result['mul_sol'] = 'Problem is infeasible (Z3 UNSAT)'

    end_time = time.time()
    result['exe_time'] = f"{int((end_time - start_time) * 1000)}ms"
    if progress_callback: progress_callback({'status': 'Finished', 'time': end_time - start_time})
    return result

# ==========================================
# PART 3: General Solver Interface
# ==========================================

class SolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables: dict[int, Any], limit: int = 5, progress_callback: Optional[Callable] = None):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.variables = variables
        self.limit = limit
        self.progress_callback = progress_callback
        self.found_solutions = []
        self.start_time: float = time.time()
        self.last_update_time: float = self.start_time
        
    def on_solution_callback(self):
        try:
            elapsed = time.time() - self.start_time
            
            # Reconstruct solution
            current_sol = []
            # Sort by step ID to ensure order
            for s in sorted(self.variables.keys()):
                val = self.Value(self.variables[s])
                current_sol.append(f"s{s+1}: u{val+1}")
            
            self.found_solutions.append(current_sol)
            
            if self.progress_callback:
                info = {
                    'solutions': len(self.found_solutions),
                    'time': elapsed,
                    'status': f'Found solution #{len(self.found_solutions)}'
                }
                self.progress_callback(info)
            
            if len(self.found_solutions) >= self.limit:
                self.StopSearch()
        except Exception:
            import traceback
            print("Callback Error:")
            traceback.print_exc()
    
    def get_stats(self) -> dict[str, Any]:
        elapsed = time.time() - self.start_time
        return {
            'solutions': len(self.found_solutions),
            'time': elapsed,
            'status': 'Searching...' if not self.found_solutions else f'Found {len(self.found_solutions)} solution(s)'
        }

def SolveInstance(filename: str, progress_callback: Optional[Callable] = None, config: Optional[dict[str, Any]] = None) -> Union[dict[str, Any], Any]:
    """
    Main Dispatcher for WSP Solvers.
    Renamed from 'Solver' to avoid conflict with z3.Solver class.
    """
    if config is None:
        config = {'algorithm': 'Auto', 'threads': 8, 'strategy': 'Default'}
    
    start_time = time.time()
    
    try:
        instance = read_file(filename)
    except Exception as e:
        return {'sat': 'error', 'sol': [], 'mul_sol': f'Parse error: {e}', 'exe_time': '0ms'}
    
    algo = config.get('algorithm', 'Auto')
    
    if algo == 'CP-SAT':
        if progress_callback:
            progress_callback({'status': 'Using CP-SAT (manual selection)...', 'time': 0})
        return Solver_CPSAT(filename, progress_callback, config)
    
    elif algo == 'Z3 SMT':
        if progress_callback:
            progress_callback({'status': 'Using Z3 SMT (manual selection)...', 'time': 0})
        try:
            return Solver_Z3(filename, progress_callback, config)
        except Exception as e:
            return {'sat': 'error', 'sol': [], 'mul_sol': f'Z3 error: {e}', 'exe_time': '0ms'}

    elif algo == 'Z3 SAT (Bool)':
        if progress_callback:
            progress_callback({'status': 'Using Z3 SAT (Bool) (manual selection)...', 'time': 0})
        try:
            return Solver_Z3_Bool(filename, progress_callback, config)
        except Exception as e:
            return {'sat': 'error', 'sol': [], 'mul_sol': f'Z3 Boolean error: {e}', 'exe_time': '0ms'}
    
    elif algo == 'Backtracking':
        if progress_callback:
            progress_callback({'status': 'Using Backtracking (manual selection)...', 'time': 0})
        try:
            return Solver_SmartBacktrack(filename, progress_callback, config)
        except Exception as e:
            return {'sat': 'error', 'sol': [], 'mul_sol': f'Backtrack error: {e}', 'exe_time': '0ms'}
    
    # Auto mode
    if instance.num_users >= 300:
        if progress_callback:
            progress_callback({'status': 'Auto: Using Smart Backtracking...', 'time': 0})
        
        try:
            result = Solver_SmartBacktrack(filename, progress_callback, config)
            if result['sat'] != 'sat' and 'z3' not in result['exe_time']:
                try:
                    if progress_callback:
                        progress_callback({'status': 'Backtrack failed, trying Z3...', 'time': time.time() - start_time})
                    return Solver_Z3(filename, progress_callback, config)
                except:
                    pass
            return result
        except Exception as e:
            if progress_callback:
                progress_callback({'status': f'Error: {e}, using Z3 SMT...', 'time': time.time() - start_time})
    
    if progress_callback:
        progress_callback({'status': 'Auto: Using Z3 SMT...', 'time': 0})
    return Solver_Z3(filename, progress_callback, config)

def Solver_CPSAT(filename: str, progress_callback: Optional[Callable] = None, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Google OR-Tools CP-SAT Solver.
    """
    start_time = time.time()
    
    try:
        instance = read_file(filename)
    except Exception as e:
        return {'sat': 'error', 'sol': [], 'mul_sol': f'Parse Error: {e}', 'exe_time': '0ms'}
    
    if progress_callback:
        progress_callback({'status': 'Analyzing...', 'time': 0})
    
    active_users = list(range(instance.num_users))
    
    if progress_callback:
        progress_callback({
            'status': 'Building model (all users)...',
            'time': time.time() - start_time
        })
    
    valid_for_step = {}
    for s in range(instance.num_steps):
        valid = [u for u in active_users if u not in instance.authorizations or s in instance.authorizations[u]]
        valid_for_step[s] = valid
        
        if not valid:
            end_time = time.time()
            return {
                'sat': 'unsat',
                'sol': [],
                'mul_sol': f'Step {s+1} has no valid users',
                'exe_time': f"{int((end_time - start_time) * 1000)}ms"
            }
    
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
    
    model = cp_model.CpModel()
    
    p = {}
    for s in range(instance.num_steps):
        p[s] = model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(valid_for_step[s]),
            f'p{s}'
        )
    
    bool_cache = {}
    def get_bool(s, u):
        if (s, u) not in bool_cache:
            b = model.NewBoolVar(f'b{s}_{u}')
            model.Add(p[s] == u).OnlyEnforceIf(b)
            model.Add(p[s] != u).OnlyEnforceIf(b.Not())
            bool_cache[(s, u)] = b
        return bool_cache[(s, u)]
    
    for s1, s2 in instance.separation_duty:
        model.Add(p[s1] != p[s2])
    
    for s1, s2 in instance.binding_duty:
        model.Add(p[s1] == p[s2])
    
    for i, (k, steps) in enumerate(instance.at_most_k):
        if k >= len(steps): continue
        relevant = set()
        for s in steps:
            relevant.update(valid_for_step[s])
        
        user_used = []
        for u in relevant:
            lits = [get_bool(s, u) for s in steps if u in valid_for_step[s]]
            if lits:
                u_var = model.NewBoolVar(f'amk{i}_u{u}')
                model.AddMaxEquality(u_var, lits)
                user_used.append(u_var)
        
        if user_used:
            model.Add(sum(user_used) <= k)
    
    for i, (steps, teams) in enumerate(instance.one_team):
        viable = []
        for t_idx, team in enumerate(teams):
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
    
    if config is None:
        config = {'timeout': 30, 'threads': 8, 'strategy': 'Default'}
    
    solver = cp_model.CpSolver()
    
    timeout = config.get('timeout', 30)
    threads = config.get('threads', 8)
    strategy = config.get('strategy', 'Default')
    
    if instance.num_users >= 400:
        solver.parameters.num_search_workers = threads if threads > 8 else 16
        solver.parameters.max_time_in_seconds = float(timeout)
        solver.parameters.linearization_level = 0
        solver.parameters.cp_model_presolve = True
        solver.parameters.cp_model_probing_level = 2
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        solver.parameters.optimize_with_core = True
        solver.parameters.optimize_with_max_hs = True
        
        if strategy == 'MinDomain':
            step_order = sorted(range(instance.num_steps), key=lambda s: len(valid_for_step[s]))
        elif strategy == 'MaxConstraint':
            step_order = sorted(range(instance.num_steps), key=lambda s: -len(valid_for_step[s]))
        else:
            step_order = sorted(range(instance.num_steps), key=lambda s: len(valid_for_step[s]))
        
        model.AddDecisionStrategy(
            [p[s] for s in step_order],
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MIN_VALUE
        )
    else:
        solver.parameters.num_search_workers = threads
        solver.parameters.max_time_in_seconds = float(timeout)
        solver.parameters.linearization_level = 0
    
    sol_limit = 5
    if config and 'solution_limit' in config:
        sol_limit = int(config['solution_limit'])

    # Enumerate solutions
    solver.parameters.enumerate_all_solutions = True
    
    callback = SolutionCallback(p, sol_limit, progress_callback)
    
    try:
        status = solver.Solve(model, callback)
    except Exception as e:
        end_time = time.time()
        return {
            'sat': 'error',
            'sol': [],
            'all_solutions': [],
            'mul_sol': f'Solver error: {e}',
            'exe_time': f"{int((end_time - start_time) * 1000)}ms"
        }
    
    result = {'sat': 'unsat', 'sol': [], 'all_solutions': [], 'mul_sol': '', 'exe_time': ''}
    
    if callback.found_solutions:
        result['sat'] = 'sat'
        result['sol'] = callback.found_solutions[0]
        result['all_solutions'] = callback.found_solutions
        
        result['mul_sol'] = f'Found {len(callback.found_solutions)} solution(s).'
        if len(callback.found_solutions) == sol_limit:
             result['mul_sol'] += " (More exist)"
        else:
             if status == cp_model.OPTIMAL: # Exhausted search
                 result['mul_sol'] += " (All found)"
             else:
                 result['mul_sol'] += " (Suboptimal/Timeout)"

    elif status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
         result['sat'] = 'error'
         result['mul_sol'] = "Error: Solver found valid assignment but callback failed to capture it."
    elif status == cp_model.INFEASIBLE:
         result['mul_sol'] = 'Problem is infeasible (CP-SAT UNSAT)'
    elif status == cp_model.UNKNOWN:
         result['mul_sol'] = 'CP-SAT timeout/unknown'
    
    end_time = time.time()
    result['exe_time'] = f"{int((end_time - start_time) * 1000)}ms"
    
    if progress_callback:
        progress_callback({'status': 'Finished', 'time': end_time - start_time})
    
    return result

def run_batch_analysis(file_list: list[str], progress_callback: Optional[Callable] = None) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """
    Runs the solver on a provided list of WSP instance files.
    """
    print(f"\n--- Starting Batch Analysis on {len(file_list)} files ---")
    print(f"{'Instance Path':<60} | {'Status':<10} | {'Time'}")
    print("-" * 85)

    stats = {'total': 0, 'sat': 0, 'unsat': 0, 'error': 0, 'timeout': 0}
    detailed_results = []
    
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        try:
            display_path = os.path.basename(file_path)
        except:
            display_path = file_path

        stats['total'] += 1
        if progress_callback:
            progress_callback({
                'current_file': display_path,
                'stats': stats.copy()
            })

        result_entry = {'file': display_path, 'status': 'UNKNOWN', 'time_ms': 0}

        try:
            result = SolveInstance(file_path)
            status = result['sat'].upper()
            time_str = result['exe_time']
            
            try:
                ms = int(time_str.replace('ms', ''))
            except:
                ms = 0
            
            result_entry['status'] = status
            result_entry['time_ms'] = ms

            if status == 'SAT': stats['sat'] += 1
            elif status == 'UNSAT': stats['unsat'] += 1
            elif status == 'ERROR':
                stats['error'] += 1
                status = "SKIP/ERR"
            else: 
                # UNKNOWN or other
                if 'timeout' in result.get('mul_sol', '').lower():
                    status = "TIMEOUT"
                stats['timeout'] += 1
            
            print(f"{display_path:<60} | {status:<10} | {time_str}")
            
        except Exception as e:
            stats['error'] += 1
            result_entry['status'] = 'ERROR'
            print(f"{display_path:<60} | CRASHED    | {str(e)}")
        
        detailed_results.append(result_entry)
        sys.stdout.flush()

    print("-" * 85)
    print(f"Batch Analysis Complete. Total: {stats['total']} | SAT: {stats['sat']} | UNSAT: {stats['unsat']} | ERR: {stats['error']}")
    
    return stats, detailed_results

# ==========================================
# PART 4: GUI Application
# ==========================================

class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state="disabled")
        self.widget.update_idletasks()

    def flush(self):
        pass

class WSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WSP Solver - Advanced Configuration")
        self.root.geometry("900x800")
        
        self.tab_solver = tk.Frame(root)
        self.tab_solver.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self._init_solver_tab()
        
        self.selected_file_path = None
        self.selected_batch_files = [] 
        self.solving = False
        self.stop_requested = False

    def _init_solver_tab(self):
        top_frame = tk.Frame(self.tab_solver, pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
        
        self.lbl_file = tk.Label(top_frame, text="No file selected", anchor="w", relief=tk.SUNKEN)
        self.lbl_file.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.btn_browse = tk.Button(top_frame, text="Browse", command=self.browse_file)
        self.btn_browse.pack(side=tk.LEFT)

        config_frame = tk.LabelFrame(self.tab_solver, text="⚙️ Solver Configuration", pady=10, padx=10, font=("Arial", 9, "bold"))
        config_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))
        
        alg_frame = tk.Frame(config_frame)
        alg_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        
        tk.Label(alg_frame, text="Algorithm:", width=12, anchor="w").pack(side=tk.LEFT, padx=(0, 5))
        
        self.var_algorithm = tk.StringVar(value="Auto")
        algorithms = ["Auto", "CP-SAT", "Z3 SMT", "Z3 SAT (Bool)", "Backtracking"]
        self.cmb_algorithm = ttk.Combobox(alg_frame, textvariable=self.var_algorithm, values=algorithms, state="readonly", width=15)
        self.cmb_algorithm.pack(side=tk.LEFT, padx=(0, 20))
        self.cmb_algorithm.bind("<<ComboboxSelected>>", self.on_algorithm_change)
        

        
        adv_frame = tk.Frame(config_frame)
        adv_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        
        tk.Label(adv_frame, text="Threads:", width=12, anchor="w").pack(side=tk.LEFT, padx=(0, 5))
        self.var_threads = tk.IntVar(value=8)
        self.scale_threads = tk.Scale(adv_frame, from_=1, to=16, orient=tk.HORIZONTAL,
                                      variable=self.var_threads, length=100, showvalue=0)
        self.scale_threads.pack(side=tk.LEFT)
        self.lbl_threads_val = tk.Label(adv_frame, text="8", width=4, anchor="w")
        self.lbl_threads_val.pack(side=tk.LEFT, padx=(0, 20))
        self.var_threads.trace('w', lambda *args: self.lbl_threads_val.config(text=str(self.var_threads.get())))
        # Search Strategy
        tk.Label(adv_frame, text="Strategy:", width=8, anchor="w").pack(side=tk.LEFT, padx=(0, 5))
        self.var_strategy = tk.StringVar(value="Default")
        strategies = ["Default", "MinDomain", "MaxConstraint", "Random"]
        self.cmb_strategy = ttk.Combobox(adv_frame, textvariable=self.var_strategy, values=strategies, state="readonly", width=12)
        self.cmb_strategy.pack(side=tk.LEFT, padx=(0, 20))

        # Solution Limit (New)
        tk.Label(adv_frame, text="Sol. Limit:", width=8, anchor="w").pack(side=tk.LEFT, padx=(0, 5))
        self.var_sol_limit = tk.IntVar(value=5)
        self.spn_sol_limit = tk.Spinbox(adv_frame, from_=1, to=100, textvariable=self.var_sol_limit, width=5)
        self.spn_sol_limit.pack(side=tk.LEFT)

        # --- Middle Frame: Actions ---
        action_frame = tk.Frame(self.tab_solver, pady=10)
        action_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
        
        self.btn_run = tk.Button(action_frame, text="Solve Instance", command=self.run_solver, bg="#dddddd", state=tk.DISABLED)
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.btn_stop = tk.Button(action_frame, text="⏸ Wait for Timeout", command=self.stop_solver, bg="#ffcccc", state=tk.DISABLED, font=("Arial", 9))
        self.btn_stop.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # --- Batch File Selection ---
        batch_frame = tk.LabelFrame(self.tab_solver, text="Batch File Selection", pady=5, padx=10)
        batch_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        btn_frame = tk.Frame(batch_frame)
        btn_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Button(btn_frame, text="📂 Select Multiple Files", command=self.select_batch_files).pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(btn_frame, text="Clear List", command=self.clear_batch_files).pack(side=tk.LEFT)
        
        self.lst_batch = tk.Listbox(batch_frame, height=5, selectmode=tk.EXTENDED)
        self.lst_batch.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(self.lst_batch)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.lst_batch.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.lst_batch.yview)

        self.btn_batch = tk.Button(batch_frame, text="Run Selected Files", command=self.run_batch, bg="#cceeff", state=tk.DISABLED)
        self.btn_batch.pack(side=tk.TOP, fill=tk.X, pady=(5,0))

        # --- Progress Frame ---
        progress_frame = tk.Frame(self.tab_solver, pady=5)
        progress_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
        
        self.lbl_progress_status = tk.Label(progress_frame, text="Ready", anchor="w", font=("Arial", 9))
        self.lbl_progress_status.pack(side=tk.TOP, fill=tk.X)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate', length=300)
        self.progress_bar.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        
        self.lbl_progress_info = tk.Label(progress_frame, text="", anchor="w", font=("Arial", 8), fg="gray")
        self.lbl_progress_info.pack(side=tk.TOP, fill=tk.X)

        # --- Bottom Frame: Output ---
        self.txt_output = scrolledtext.ScrolledText(self.tab_solver, state='disabled', font=("Consolas", 10))
        self.txt_output.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    


    def on_algorithm_change(self, event=None):
        algo = self.var_algorithm.get()
        if algo == "Z3 SMT":
            self.lbl_progress_status.config(text="Z3 SMT Solver selected - Good for complex constraints", fg="blue")
        elif algo == "Z3 SAT (Bool)":
            self.lbl_progress_status.config(text="Z3 SAT - Uses pure boolean logic (Alternative Formulation)", fg="purple")
        elif algo == "Backtracking":
            self.lbl_progress_status.config(text="Backtracking selected - Good for large instances", fg="blue")
        elif algo == "CP-SAT":
            self.lbl_progress_status.config(text="CP-SAT selected - Google's constraint solver", fg="blue")
        else:
            self.lbl_progress_status.config(text="Auto mode - Will choose best algorithm automatically", fg="black")

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
        self.stop_requested = False
        self.btn_run.config(state=tk.DISABLED)
        self.btn_batch.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        
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
        
        t = threading.Thread(target=self._solve_thread)
        t.daemon = True
        t.start()
    
    def stop_solver(self):
        if self.solving:
            self.stop_requested = True
            self.progress_bar.stop()
            self.progress_bar['mode'] = 'determinate'
            self.progress_bar['value'] = 50
            timeout = self.var_timeout.get()
            self.lbl_progress_status.config(text=f"⏹ Stop requested - Waiting for timeout ({timeout}s)...", fg="orange")
            self.btn_stop.config(state=tk.DISABLED)
            self._safe_print(f"\n⏹ STOP REQUESTED. Waiting for timeout ({timeout}s)...\n")
    
    def _update_progress(self, info):
        def update():
            status = info.get('status', 'Running...')
            if 'Error' in status or 'error' in status.lower():
                self.lbl_progress_status.config(text=status, fg="red")
            elif 'Finished' in status:
                self.lbl_progress_status.config(text=status, fg="green")
            elif 'Found solution' in status:
                self.lbl_progress_status.config(text=status, fg="blue")
            else:
                self.lbl_progress_status.config(text=status, fg="black")
            
            info_parts = []
            if 'num_vars' in info: info_parts.append(f"Variables: {info['num_vars']}")
            if 'solutions' in info and info['solutions'] > 0: info_parts.append(f"Solutions: {info['solutions']}")
            if 'time' in info and info['time'] is not None and isinstance(info['time'], (int, float)):
                 info_parts.append(f"Time: {info['time']:.2f}s")
            
            if info_parts:
                self.lbl_progress_info.config(text=" | ".join(info_parts))
            
            if status == 'Finished':
                self.progress_bar.stop()
                self.progress_bar['mode'] = 'determinate'
                self.progress_bar['value'] = 100
            elif 'Found solution' in status and self.progress_bar['mode'] != 'indeterminate':
                self.progress_bar['mode'] = 'indeterminate'
                self.progress_bar.start(10)
        
        self.root.after(0, update)
        
    def _solve_thread(self):
        try:
            config = {
                'algorithm': self.var_algorithm.get(),
                'threads': self.var_threads.get(),
                'strategy': self.var_strategy.get(),
                'solution_limit': self.var_sol_limit.get()
            }
            
            self._safe_print(f"\n⚙️ Configuration: {config['algorithm']}, Limit: {config['solution_limit']}\n")
            
            # Using SolveInstance renamed function
            result = SolveInstance(self.selected_file_path, progress_callback=self._update_progress, config=config)
            
            self._update_progress({'status': 'Finished', 'time': float(result['exe_time'].replace('ms', '')) / 1000})
            
            output_text = "\n--- Results ---\n"
            output_text += f"Status: {result['sat']}\n"
            output_text += f"Time:   {result['exe_time']}\n"
            
            if result['sat'] == 'sat':
                solutions = result.get('all_solutions', [result['sol']])
                # Normalize if backend sends simple list
                if solutions and isinstance(solutions[0], str):
                    solutions = [solutions]

                output_text += f"\n✅ Found {len(solutions)} solution(s) (Limit: {config['solution_limit']}):\n"
                for i, sol in enumerate(solutions):
                    output_text += f"\n[Solution {i+1}]\n"
                    # Format: 4 items per line
                    for j in range(0, len(sol), 4):
                         output_text += "  " + ", ".join(sol[j:j+4]) + "\n"
                
                if result.get('mul_sol'):
                    output_text += f"\n{result['mul_sol']}\n"

            elif result['sat'] == 'error':
                output_text += f"Error: {result.get('mul_sol', 'Unknown error')}\n"
            else:
                output_text += f"No solution found. ({result.get('mul_sol', '')})\n"
                  
            self._safe_print(output_text)
            
        except Exception as e:
            self._update_progress({'status': 'Error', 'time': 0})
            self._safe_print(f"Error: {str(e)}\n")
            import traceback
            traceback.print_exc()
        finally:
            self.solving = False
            self.stop_requested = False
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_batch.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_stop.config(state=tk.DISABLED))

    def select_batch_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if files:
            for f in files:
                if f not in self.selected_batch_files:
                    self.selected_batch_files.append(f)
                    self.lst_batch.insert(tk.END, os.path.basename(f))
            self.btn_batch.config(text=f"Run Selected ({len(self.selected_batch_files)})", state=tk.NORMAL)

    def clear_batch_files(self):
        self.selected_batch_files = []
        self.lst_batch.delete(0, tk.END)
        self.btn_batch.config(text="Run Selected Files", state=tk.DISABLED)

    def run_batch(self):
        if not self.selected_batch_files:
            tk.messagebox.showwarning("No Files", "Please select files to run.")
            return

        if self.solving: return
        
        self.solving = True
        self.btn_run.config(state=tk.DISABLED)
        self.btn_batch.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.DISABLED) # Batch stop not implemented cleanly
        
        self.txt_output.configure(state='normal')
        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, f"Starting Batch Analysis on {len(self.selected_batch_files)} files...\n")
        self.txt_output.configure(state='disabled')
        
        t = threading.Thread(target=self._batch_thread)
        t.daemon = True
        t.start()

    def _batch_progress_callback(self, info):
        def update():
            if 'current_file' in info:
                self.lbl_progress_status.config(text=f"Processing: {info['current_file']}", fg="blue")
            
            if 'stats' in info:
                stats = info['stats']
                self.lbl_progress_info.config(text=f"Total: {stats['total']} | SAT: {stats['sat']} | UNSAT: {stats['unsat']} | ERR: {stats['error']}")
                self.progress_bar['mode'] = 'determinate'
                self.progress_bar['maximum'] = len(self.selected_batch_files)
                self.progress_bar['value'] = stats['total']
        
        self.root.after(0, update)

    def _batch_thread(self):
        old_stdout = sys.stdout
        sys.stdout = TextRedirector(self.txt_output)
        
        try:
            stats, detailed = run_batch_analysis(self.selected_batch_files, progress_callback=self._batch_progress_callback)
        finally:
            sys.stdout = old_stdout
            
        def on_complete():
            self.solving = False
            self.btn_run.config(state=tk.NORMAL)
            self.btn_batch.config(state=tk.NORMAL, text=f"Run Selected ({len(self.selected_batch_files)})")
            self.lbl_progress_status.config(text="Batch Analysis Complete", fg="green")
            self.progress_bar['value'] = 0
            
            # Print summary to output
            self.txt_output.configure(state='normal')
            self.txt_output.insert(tk.END, "\nBatch Analysis Complete.\n")
            self.txt_output.insert(tk.END, f"Total: {stats['total']} | SAT: {stats['sat']} | UNSAT: {stats['unsat']} | ERR: {stats['error']}\n")
            self.txt_output.configure(state='disabled')
        
        self.root.after(0, on_complete)

            
    def _safe_print(self, text):
        self.root.after(0, lambda: self._append_text(text))
        
    def _append_text(self, text):
        self.txt_output.configure(state='normal')
        self.txt_output.insert(tk.END, text)
        self.txt_output.see(tk.END)
        self.txt_output.configure(state='disabled')

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        run_batch_analysis()
    else:
        root = tk.Tk()
        app = WSPApp(root)
        root.mainloop()