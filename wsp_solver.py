
import time
from ortools.sat.python import cp_model
from wsp_parser import read_file, Instance

def Solver(filename):
    """
    Solves the WSP instance defined in filename.
    Returns dictionary with keys: 'sat', 'sol', 'exe_time'.
    Strictly forbids printing inside.
    """
    start_time = time.time()
    
    try:
        instance = read_file(filename)
    except Exception as e:
        end_time = time.time()
        exe_time = int((end_time - start_time) * 1000)
        return {'sat': 'error', 'sol': [], 'exe_time': f'{exe_time}ms'}

    # Pre-processing: Identify "Super Users" and valid users for each step
    super_users = [u for u in range(instance.num_users) if u not in instance.authorizations]

    # Map step -> set of valid users (Optimization: strictly authorized + super users)
    # Note: instance.authorizations is u -> [s1, s2...]
    # We want s -> [u1, u2...]
    step_to_users = {s: set(super_users) for s in range(instance.num_steps)}
    for u, steps in instance.authorizations.items():
        for s in steps:
            step_to_users[s].add(u)

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
    # Solve
    solver = cp_model.CpSolver()
    
    # Final Performance Tuning
    # Use 8 workers (or all distinct cores)
    solver.parameters.num_search_workers = 8
    # 0 = No linearization (often better for boolean/logical constraints)
    solver.parameters.linearization_level = 0
    # Ensure presolve is enabled (usually default, but explicit for clarity)
    solver.parameters.cp_model_presolve = True

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
