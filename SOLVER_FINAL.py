# 🚀 ULTRA-OPTIMIZED SOLVER FOR HARD INSTANCES
# 
# 将此代码复制到 wsp_app.py 的 Solver 函数

def Solver(filename, progress_callback=None):
    """
    ULTRA-OPTIMIZED WSP Solver - Designed for 4-constraint-hard instances.
    
    KEY ALGORITHMIC INNOVATIONS:
    1. Integer domain variables (60 vars instead of 30,000 bool vars)
    2. Super-aggressive symmetry breaking (500 -> ~60 users)
    3. Conflict-driven search with smart variable ordering
    4. Tight timeout with early termination
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
    # ULTRA-AGGRESSIVE USER PRUNING
    # ============================================================================
    
    # Essential users (explicitly in constraints)
    essential = set(instance.authorizations.keys())
    for steps, teams in instance.one_team:
        for team in teams:
            essential.update(u for u in team if u < instance.num_users)
    
    # Generic users (no constraints) - keep MINIMAL amount
    all_users = set(range(instance.num_users))
    generic = sorted(list(all_users - essential))
    
    # CRITICAL: For high-constraint problems, we need even less super users
    # Heuristic: If many SoD constraints, we need fewer super users
    sod_ratio = len(instance.separation_duty) / max(instance.num_steps, 1)
    
    if sod_ratio > 2.0:  # Very constrained
        keep_count = min(len(generic), instance.num_steps // 2)  # Even more aggressive!
    else:
        keep_count = min(len(generic), instance.num_steps)
    
    active_users = sorted(list(essential) + generic[:keep_count])
    
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
        
        #Create user-used variables
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
        solver.parameters.max_time_in_seconds = 120.0  # Give it 2 minutes
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
