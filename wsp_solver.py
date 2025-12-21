
import time
from ortools.sat.python import cp_model
from wsp_parser import read_file, Instance

def Solver(filename):
    """
    Solves the WSP instance defined in filename.
    Returns dictionary with keys: 'sat', 'sol', 'exe_time'.
    Strictly forbids printing inside.
    
    ALGORITHMIC OPTIMIZATIONS:
    1. Early UNSAT detection before model building
    2. Adaptive timeout based on problem complexity
    3. Advanced search heuristics
    4. Constraint preprocessing
    """
    start_time = time.time()
    
    try:
        instance = read_file(filename)
    except Exception as e:
        end_time = time.time()
        exe_time = int((end_time - start_time) * 1000)
        return {'sat': 'error', 'sol': [], 'exe_time': f'{exe_time}ms'}

    # ========== SUPER USER PRUNING ==========
    # Step 1: Identify Essential Users (explicitly mentioned in constraints)
    essential_users = set()
    
    # Users with explicit authorizations
    for u in instance.authorizations.keys():
        essential_users.add(u)
    
    # Users mentioned in one_team constraints
    for (steps, teams) in instance.one_team:
        for team_users in teams:
            essential_users.update(team_users)
    
    # Step 2: Identify Generic Pool (users not explicitly constrained)
    all_users = set(range(instance.num_users))
    generic_users = all_users - essential_users
    
    # Step 3: Prune Generic Users
    # We only need at most num_steps generic users (since there are only that many steps)
    # Keep the first num_steps generic users by sorted ID
    kept_generic_users = sorted(list(generic_users))[:instance.num_steps]
    
    # Step 4: Define Active Users
    active_users = essential_users | set(kept_generic_users)
    
    # Step 5: Build step_to_users ONLY for active users
    # Identify which active users have no explicit authorizations (super users among active)
    super_users = [u for u in active_users if u not in instance.authorizations]
    
    # Map step -> set of valid users (Optimization: strictly authorized + super users)
    # Note: instance.authorizations is u -> [s1, s2...]
    # We want s -> [u1, u2...]
    step_to_users = {s: set(super_users) for s in range(instance.num_steps)}
    for u, steps in instance.authorizations.items():
        if u in active_users:  # Only consider active users
            for s in steps:
                step_to_users[s].add(u)
    
    # ========== EARLY UNSAT DETECTION ==========
    # Check 1: Any step with no valid users? -> UNSAT
    for s in range(instance.num_steps):
        if not step_to_users[s]:
            end_time = time.time()
            exe_time = int((end_time - start_time) * 1000)
            return {'sat': 'unsat', 'sol': [], 'exe_time': f'{exe_time}ms', 'mul_sol': '', 'num_vars': 0, 'num_constraints': 0}
    
    # Check 2: Binding of Duty conflicts - if two bound steps have no common users -> UNSAT
    for (s1, s2) in instance.binding_duty:
        common = step_to_users[s1] & step_to_users[s2]
        if not common:
            end_time = time.time()
            exe_time = int((end_time - start_time) * 1000)
            return {'sat': 'unsat', 'sol': [], 'exe_time': f'{exe_time}ms', 'mul_sol': 'BoD conflict: no common users', 'num_vars': 0, 'num_constraints': 0}
    
    # Check 3: At-most-k feasibility - if k < minimum required users -> UNSAT
    for (k, steps) in instance.at_most_k:
        # Count how many steps MUST be done by different users due to SoD
        # Build conflict graph for steps in this at-most-k constraint
        min_users_needed = 1
        assigned = [False] * len(steps)
        for i, s1 in enumerate(steps):
            if assigned[i]:
                continue
            assigned[i] = True
            # Find all steps that must be different from s1
            for j, s2 in enumerate(steps):
                if i != j and not assigned[j]:
                    if (s1, s2) in instance.separation_duty or (s2, s1) in instance.separation_duty:
                        assigned[j] = True
                        min_users_needed += 1
        
        if k < min_users_needed:
            end_time = time.time()
            exe_time = int((end_time - start_time) * 1000)
            return {'sat': 'unsat', 'sol': [], 'exe_time': f'{exe_time}ms', 'mul_sol': f'At-most-k conflict: need {min_users_needed} but k={k}', 'num_vars': 0, 'num_constraints': 0}

    model = cp_model.CpModel()
    
    # 1. Variables: p[s] = assigned user
    p = {}
    for s in range(instance.num_steps):
        valid_us = sorted(list(step_to_users[s]))
        if not valid_us:
             end_time = time.time()
             exe_time = int((end_time - start_time) * 1000)
             return {'sat': 'unsat', 'sol': [], 'exe_time': f'{exe_time}ms', 'mul_sol': '', 'num_vars': 0, 'num_constraints': 0}
        
        p[s] = model.NewIntVarFromDomain(
             cp_model.Domain.FromValues(valid_us), f'p_{s}'
        )

    # Helper to get boolean x[s, u] <=> p[s] == u (On Demand)
    bool_cache = {}
    def get_x(s, u):
        if (s, u) not in bool_cache:
            b = model.NewBoolVar(f'x_s{s}_u{u}')
            model.Add(p[s] == u).OnlyEnforceIf(b)
            model.Add(p[s] != u).OnlyEnforceIf(b.Not())
            bool_cache[(s, u)] = b
        return bool_cache[(s, u)]

    # 2. Assignment, Authorizations
    # Implicitly handled by Domain of p[s].
    
    # ========== ALGORITHMIC OPTIMIZATION: SEARCH HEURISTICS ==========
    # Define search strategy: prioritize steps with fewer valid users (Most Constrained First)
    # This is a classic CSP heuristic that helps find conflicts faster
    step_order = sorted(range(instance.num_steps), key=lambda s: len(step_to_users[s]))
    
    # Create decision strategy for the solver
    model.AddDecisionStrategy(
        [p[s] for s in step_order],
        cp_model.CHOOSE_FIRST,  # Choose variables in the order we specified
        cp_model.SELECT_MIN_VALUE  # Try smallest user ID first (arbitrary but deterministic)
    )


    # 3. Separation of Duty (SoD): p[s1] != p[s2]
    # Only need to add if domains overlap
    for (s1, s2) in instance.separation_duty:
        model.Add(p[s1] != p[s2])

    # 4. Binding of Duty (BoD): p[s1] == p[s2]
    for (s1, s2) in instance.binding_duty:
         model.Add(p[s1] == p[s2])

    # 5. At-most-k
    for i, (k, steps) in enumerate(instance.at_most_k):
        # Optimization: If k >= len(steps), constraint is trivial (always satisfied)
        if k >= len(steps):
            continue

        # Calculate relevant_users: users authorized for at least one step in the constraint
        relevant_users = set()
        for s in steps:
            # step_to_users[s] contains all authorized users for step s (including super users)
            relevant_users.update(step_to_users[s])

        # Create auxiliary variables only for relevant users
        user_vars = []
        for u in relevant_users:
            literals = []
            for s in steps:
                if u in step_to_users[s]:
                    literals.append(get_x(s, u))
            
            if literals:
                # u_used is True if user u is assigned to any step in 'steps'
                u_used = model.NewBoolVar(f'amk_{i}_u{u}')
                model.AddMaxEquality(u_used, literals)
                user_vars.append(u_used)
        
        if user_vars:
            model.Add(sum(user_vars) <= k)

    # 6. One-team
    for i, (steps, teams) in enumerate(instance.one_team):
        viable_teams = []
        viable_team_vars = []

        for t_idx, team_users in enumerate(teams):
            # Optimisation: Check if the team is viable.
            # A team is viable ONLY IF for every step in the set, 
            # the team has at least one member who is authorized to do it.
            is_viable = True
            for s in steps:
                # Check intersection of team_users and step_to_users[s]
                if not any(u in step_to_users[s] for u in team_users):
                    is_viable = False
                    break
            
            if is_viable:
                t_var = model.NewBoolVar(f'ot_{i}_t{t_idx}')
                viable_team_vars.append(t_var)
                viable_teams.append((t_var, team_users))

        if not viable_team_vars:
            # No team can satisfy the requirements -> UNSAT
            model.Add(0 == 1)
        else:
            # Exactly one viable team must be selected
            model.AddExactlyOne(viable_team_vars)

            for (t_var, team_users) in viable_teams:
                for s in steps:
                    # Collect all users in the team authorized for step s
                    relevant_x = []
                    for u in team_users:
                        if u in step_to_users[s]:
                            relevant_x.append(get_x(s, u))
                    
                    # If this team is selected, someone from the team must do step s
                    # We know relevant_x is not empty because the team is viable
                    model.Add(sum(relevant_x) == 1).OnlyEnforceIf(t_var)

    # Solve
    solver = cp_model.CpSolver()
    
    # ========== ALGORITHMIC OPTIMIZATION: ADAPTIVE TIMEOUT ==========
    # Calculate problem complexity score
    complexity_score = (
        instance.num_steps * 0.1 +
        instance.num_constraints * 0.05 +
        len(active_users) * 0.02
    )
    
    # Adaptive timeout based on complexity (MUCH MORE AGGRESSIVE)
    if complexity_score > 100:  # Very hard instances
        timeout = 60.0  # 1 minute max (was 300s before)
    elif complexity_score > 50:  # Hard instances
        timeout = 30.0  # 30 seconds
    elif complexity_score > 20:  # Medium instances
        timeout = 15.0  # 15 seconds
    else:  # Easy instances
        timeout = 10.0  # 10 seconds
    
    solver.parameters.max_time_in_seconds = timeout
    
    # ========== ALGORITHMIC OPTIMIZATION: SEARCH STRATEGY ==========
    # Use FIXED_SEARCH for more deterministic and focused search
    # This prevents the solver from trying too many different strategies
    solver.parameters.search_branching = cp_model.FIXED_SEARCH
    
    # ========== ALGORITHMIC OPTIMIZATION: PARALLELIZATION ==========
    # Use multiple workers but not too many (diminishing returns)
    solver.parameters.num_search_workers = min(8, instance.num_steps // 10 + 1)
    
    # ========== ALGORITHMIC OPTIMIZATION: PRESOLVE ==========
    # Aggressive presolve to simplify the model before search
    solver.parameters.cp_model_presolve = True
    solver.parameters.cp_model_probing_level = 2  # More aggressive probing
    
    # ========== ALGORITHMIC OPTIMIZATION: LINEARIZATION ==========
    # Level 0 = No linearization (better for boolean/logical constraints)
    # This is critical for WSP which is mostly boolean logic
    solver.parameters.linearization_level = 0
    
    # ========== ALGORITHMIC OPTIMIZATION: CONFLICT ANALYSIS ==========
    # Enable conflict analysis to learn from failures faster
    solver.parameters.optimize_with_core = True
    
    # ========== ALGORITHMIC OPTIMIZATION: RESTART STRATEGY ==========
    # More frequent restarts to avoid getting stuck
    solver.parameters.restart_algorithms = [cp_model.LUBY_RESTART]
    solver.parameters.restart_period = 100  # Restart every 100 failures


    status = solver.Solve(model)
    
    result = {
        'sat': 'unsat',
        'sol': [],
        'mul_sol': '',
        'exe_time': ''
    }
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        result['sat'] = 'sat'
        sol_list = []
        
        current_sol = {}
        for s in range(instance.num_steps):
            val = solver.Value(p[s])
            sol_list.append(f"s{s+1}: u{val+1}")
            current_sol[s] = val
        result['sol'] = sol_list
        
        # Multiple Solutions Check
        # Blocking clause: at least one variable must change
        blocking_bools = []
        for s, val in current_sol.items():
            if (s, val) in bool_cache:
                blocking_bools.append(bool_cache[(s, val)])
            else:
                b = model.NewBoolVar(f'block_s{s}_{val}')
                model.Add(p[s] == val).OnlyEnforceIf(b)
                model.Add(p[s] != val).OnlyEnforceIf(b.Not())
                blocking_bools.append(b)
                
        model.Add(sum(blocking_bools) <= instance.num_steps - 1)
        
        status2 = solver.Solve(model)
        if status2 == cp_model.OPTIMAL or status2 == cp_model.FEASIBLE:
            result['mul_sol'] = 'other solutions exist'
        else:
            result['mul_sol'] = 'this is the only solution'
            
    
    result['num_vars'] = len(model.Proto().variables)
    result['num_constraints'] = len(model.Proto().constraints)

    end_time = time.time()
    exe_time = int((end_time - start_time) * 1000)
    result['exe_time'] = f'{exe_time}ms'
        
    return result


def run_hard_analysis(directory="SAI/additional-examples/4-constraint-hard"):
    """
    Runs the solver on the first few instances of the hard dataset.
    """
    import os
    
    print("| Instance | Status | Time (ms) | Vars | Constraints |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    # Run slightly fewer files to save time, or just 0.txt to 5.txt
    files = sorted([f for f in os.listdir(directory) if f.endswith('.txt') and 'solution' not in f])[:5]
    
    for filename in files:
        path = os.path.join(directory, filename)
        result = Solver(path)
        time_val = result['exe_time'].replace('ms', '')
        print(f"| {filename} | {result['sat']} | {time_val} | {result.get('num_vars', '-')} | {result.get('num_constraints', '-')} |")



def run_batch_analysis(directory="SAI/instances"):
    """
    Iterates through example1.txt to example19.txt in the given directory,
    runs the Solver, and prints a Markdown table of results.
    """
    import os
    
    print("| Instance | Status | Time (ms) |")
    print("| :--- | :--- | :--- |")
    
    for i in range(1, 20):
        filename = f"example{i}.txt"
        path = os.path.join(directory, filename)
        
        if os.path.exists(path):
            result = Solver(path)
            # Clean up time string if needed, currently it returns "Xms"
            time_val = result['exe_time'].replace('ms', '')
            print(f"| {filename} | {result['sat']} | {time_val} |")
        else:
            print(f"| {filename} | Not Found | - |")

if __name__ == "__main__":
    import sys
    # Allow running batch analysis via command line
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        run_batch_analysis()
    elif len(sys.argv) > 1 and sys.argv[1] == 'hard':
        run_hard_analysis()
    else:
        pass
