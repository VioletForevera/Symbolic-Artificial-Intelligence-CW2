
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

    model = cp_model.CpModel()
    
    # Decision Variables: x[s, u]
    x = {}
    for s in range(instance.num_steps):
        for u in range(instance.num_users):
            x[s, u] = model.NewBoolVar(f'x_s{s}_u{u}')
            
    # 1. Assignment Constraint (Exactly One User per Step)
    for s in range(instance.num_steps):
        model.AddExactlyOne([x[s, u] for u in range(instance.num_users)])
        
    # 2. Authorization Constraint
    # Rule: If user is in dict, they are restricted to those steps.
    # If user NOT in dict, they are free (no constraints added).
    for u in range(instance.num_users):
        if u in instance.authorizations:
            allowed_steps = set(instance.authorizations[u])
            for s in range(instance.num_steps):
                if s not in allowed_steps:
                    model.Add(x[s, u] == 0)
    
    # 3. Separation of Duty (SoD)
    for (s1, s2) in instance.separation_duty:
        for u in range(instance.num_users):
            model.Add(x[s1, u] + x[s2, u] <= 1)
            
    # 4. Binding of Duty (BoD)
    for (s1, s2) in instance.binding_duty:
        for u in range(instance.num_users):
            model.Add(x[s1, u] == x[s2, u])

    # 5. At-most-k
    for i, (k, steps) in enumerate(instance.at_most_k):
        user_used = []
        for u in range(instance.num_users):
            u_var = model.NewBoolVar(f'amk_{i}_u{u}')
            # u_var implies user u performs at least one step in `steps`
            model.AddMaxEquality(u_var, [x[s, u] for s in steps])
            user_used.append(u_var)
        model.Add(sum(user_used) <= k)

    # 6. One-team
    for i, (steps, teams) in enumerate(instance.one_team):
        team_vars = [model.NewBoolVar(f'ot_{i}_t{t}') for t in range(len(teams))]
        model.AddExactlyOne(team_vars)
        
        for t_idx, team_users in enumerate(teams):
            # If team selected, then for each step s, the assigned user MUST be in team_users.
            # Implies sum(x[s, u] for u in team_users) == 1
            for s in steps:
                valid_users = [x[s, u] for u in team_users if u < instance.num_users]
                model.Add(sum(valid_users) == 1).OnlyEnforceIf(team_vars[t_idx])

    # Solve
    solver = cp_model.CpSolver()
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
        assigned_vars = []
        
        for s in range(instance.num_steps):
            for u in range(instance.num_users):
                if solver.Value(x[s, u]):
                    sol_list.append(f"s{s+1}: u{u+1}")
                    assigned_vars.append(x[s, u])
                    break
        result['sol'] = sol_list
        
        # Multiple Solutions Check
        # Add blocking clause: sum of currently true variables must be < num_steps
        model.Add(sum(assigned_vars) <= instance.num_steps - 1)
        
        status2 = solver.Solve(model)
        if status2 == cp_model.OPTIMAL or status2 == cp_model.FEASIBLE:
            result['mul_sol'] = 'other solutions exist'
        else:
            result['mul_sol'] = 'this is the only solution'
            
    end_time = time.time()
    exe_time = int((end_time - start_time) * 1000)
    result['exe_time'] = f'{exe_time}ms'
        
    return result


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
    else:
        pass
