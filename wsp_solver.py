
import time
from ortools.sat.python import cp_model
from wsp_parser import read_file, Instance

def solve_wsp(filename):
    """
    Parses a WSP instance from a file, solves it using CP-SAT,
    handles advanced constraints, and returns a formatted dictionary.
    """
    # 1. Parse the instance
    try:
        instance = read_file(filename)
    except Exception as e:
        return {'sat': 'error', 'sol': [], 'mul_sol': str(e), 'exe_time': '0ms'}

    model = cp_model.CpModel()
    
    # 2. Decision Variables
    # x[s, u] is True if step s is assigned to user u
    x = {}
    for s in range(instance.num_steps):
        for u in range(instance.num_users):
            x[s, u] = model.NewBoolVar(f'x_s{s}_u{u}')
            
    # 3. Base Constraint: Every step must be assigned to exactly one user
    for s in range(instance.num_steps):
        model.AddExactlyOne([x[s, u] for u in range(instance.num_users)])
        
    # 4. Content-based Constraints (Auth, SoD, BoD)
    
    # Authorizations
    for u in range(instance.num_users):
        if u in instance.authorizations:
            allowed_steps = set(instance.authorizations[u])
            for s in range(instance.num_steps):
                if s not in allowed_steps:
                    model.Add(x[s, u] == 0)

    # Separation of Duty (SoD)
    for (s1, s2) in instance.separation_duty:
        for u in range(instance.num_users):
            model.Add(x[s1, u] + x[s2, u] <= 1)
            
    # Binding of Duty (BoD)
    for (s1, s2) in instance.binding_duty:
        for u in range(instance.num_users):
            model.Add(x[s1, u] == x[s2, u])

    # 5. Advanced Constraints (At-most-k, One-team)

    # At-most-k
    # Constraint: For a set of steps T, at most k distinct users.
    for i, (k, steps) in enumerate(instance.at_most_k):
        # Aux variable: user_used[u] is true if user u performs ANY step in the set 'steps'
        user_used = []
        for u in range(instance.num_users):
            u_var = model.NewBoolVar(f'amk_{i}_user_{u}_used')
            # u_var <=> (x[s1, u] OR x[s2, u] OR ... for s in steps)
            # Implemented as: u_var = max(x[s, u] for s in steps)
            # Since x are booleans, max is equivalent to OR
            vars_for_user_in_steps = [x[s, u] for s in steps]
            model.AddMaxEquality(u_var, vars_for_user_in_steps)
            user_used.append(u_var)
        
        # Constraint: Sum of used users <= k
        model.Add(sum(user_used) <= k)

    # One-team
    # Constraint: Steps T must be assigned to users in exactly one of the provided teams.
    for i, (steps, teams) in enumerate(instance.one_team):
        # team_selected[t_idx] is true if team t_idx is the chosen one
        team_vars = [model.NewBoolVar(f'ot_{i}_team_{t_idx}') for t_idx in range(len(teams))]
        
        # Ensure exactly one team is selected
        model.AddExactlyOne(team_vars)
        
        for t_idx, team_users in enumerate(teams):
            # If team t_idx is selected, then for every step s in T, 
            # the assigned user MUST be in team_users.
            # Which means sum(x[s, u] for u in team_users) == 1 (since exactly one user per step)
            
            # Optimization: Pre-calculate set for fast lookup logic if needed, 
            # but here we iterate constraints.
            
            for s in steps:
                # Use OnlyEnforceIf to conditionalize the constraint
                allowed_users_vars = [x[s, u] for u in team_users if u < instance.num_users] 
                # Note: 'if u < instance.num_users' is safety, though parser ensures valid IDs.
                
                # If team is selected, one of the team members must do step s.
                model.Add(sum(allowed_users_vars) == 1).OnlyEnforceIf(team_vars[t_idx])

    # 6. Solve
    solver = cp_model.CpSolver()
    start_time = time.time()
    status = solver.Solve(model)
    end_time = time.time()
    
    exe_time = int((end_time - start_time) * 1000) # ms
    
    result = {
        'sat': 'unsat',
        'sol': [],
        'mul_sol': 'this is the only solution', # default
        'exe_time': f'{exe_time}ms'
    }
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        result['sat'] = 'sat'
        
        # Format solution: s1: u2 (1-based)
        sol_list = []
        assignment = {} # Store for blocking clause (s, u) tuples
        
        for s in range(instance.num_steps):
            for u in range(instance.num_users):
                if solver.Value(x[s, u]):
                    sol_list.append(f"s{s+1}: u{u+1}")
                    assignment[s] = u
                    break
        result['sol'] = sol_list
        
        # Check for multiple solutions
        # Create a blocking clause: prevent the exact same assignment
        # sum(x[s, assigned_u]) <= num_steps - 1
        current_sol_vars = [x[s, assignment[s]] for s in range(instance.num_steps)]
        model.Add(sum(current_sol_vars) <= instance.num_steps - 1)
        
        status_make_2 = solver.Solve(model)
        if status_make_2 == cp_model.OPTIMAL or status_make_2 == cp_model.FEASIBLE:
            result['mul_sol'] = 'other solutions exist'
        else:
            result['mul_sol'] = 'this is the only solution'
            
    return result

def Solver(filename):
    # Wrapper to match the requested function name exactly
    return solve_wsp(filename)

if __name__ == "__main__":
    # Test Block
    import sys
    import os

    # Create a complex test file
    test_content = """#Steps: 5, #Users: 4, #Constraints: 5
Authorisations u1 s1 s2
Authorisations u2 s2 s3
Separation-of-duty s1 s2
Binding-of-duty s2 s3
At-most-k 2 s1 s2 s3
One-team s4 s5 (u3) (u4)
"""
    # Explanation of test:
    # s1: u1 (must be u1, as s1 is SoD with s2(u2), u1 only does s1,s2. u2 only s2,s3)
    # s2: u2 (needed for SoD with s1)
    # s3: u2 (BoD with s2)
    # s4, s5: Must be u3 or u4 (One-team). 
    # If Team 1 (u3) -> s4=u3, s5=u3. 
    # If Team 2 (u4) -> s4=u4, s5=u4. 
    # Result: s1:u1, s2:u2, s3:u2, s4:u3, s5:u3 is valid.
    
    with open("complex_test.wsp", "w") as f:
        f.write(test_content)
    
    print("Solving complex_test.wsp...")
    res = Solver("complex_test.wsp")
    print(res)
    
    # Clean up
    if os.path.exists("complex_test.wsp"):
        os.remove("complex_test.wsp")
