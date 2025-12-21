"""
全新算法：基于专业SAT求解器的WSP求解器
使用 Glucose4/MiniSat 而不是 CP-SAT

这种方法更适合高约束密度的WSP问题
"""

import time
from pysat.solvers import Glucose4
from pysat.formula import CNF

def create_variable_mapping(num_steps, num_users):
    """创建变量映射：x[s,u] -> SAT变量ID"""
    var_map = {}
    var_id = 1
    for s in range(num_steps):
        for u in range(num_users):
            var_map[(s, u)] = var_id
            var_id += 1
    return var_map, var_id - 1

def get_var(var_map, s, u):
    """获取SAT变量ID"""
    return var_map.get((s, u), None)

def Solver_SAT(filename, progress_callback=None):
    """
    SAT-based WSP Solver using Glucose4
    
    编码方式：
    - 变量 x[s,u]: 步骤s分配给用户u
    - 使用CNF子句编码所有约束
    """
    from wsp_app import read_file  # 导入解析器
    
    start_time = time.time()
    
    # 解析实例
    try:
        instance = read_file(filename)
    except Exception as e:
        return {'sat': 'error', 'sol': [], 'mul_sol': f'Parse error: {e}', 'exe_time': '0ms'}
    
    if progress_callback:
        progress_callback({'status': 'Encoding to SAT...', 'time': 0})
    
    # 创建变量映射
    var_map, max_var = create_variable_mapping(instance.num_steps, instance.num_users)
    
    # 创建CNF公式
    cnf = []
    
    # ========================================================================
    # 约束1: 每个步骤必须有恰好一个用户 (AMO + ALO)
    # ========================================================================
    for s in range(instance.num_steps):
        # At-Least-One: 至少一个用户
        clause = []
        for u in range(instance.num_users):
            # 检查授权
            if u in instance.authorizations:
                if s in instance.authorizations[u]:
                    clause.append(get_var(var_map, s, u))
            else:
                # 超级用户
                clause.append(get_var(var_map, s, u))
        
        if not clause:
            # 没有有效用户 -> UNSAT
            return {'sat': 'unsat', 'sol': [], 'mul_sol': 'No valid users for step', 'exe_time': '0ms'}
        
        cnf.append(clause)
        
        # At-Most-One: 最多一个用户（使用pairwise encoding）
        valid_users = [u for u in range(instance.num_users) 
                       if (u not in instance.authorizations or s in instance.authorizations[u])]
        
        for i, u1 in enumerate(valid_users):
            for u2 in valid_users[i+1:]:
                # ¬x[s,u1] ∨ ¬x[s,u2]
                cnf.append([-get_var(var_map, s, u1), -get_var(var_map, s, u2)])
    
    # ========================================================================
    # 约束2: Separation of Duty
    # ========================================================================
    for s1, s2 in instance.separation_duty:
        # 找出能同时做s1和s2的用户
        for u in range(instance.num_users):
            can_do_s1 = (u not in instance.authorizations or s1 in instance.authorizations[u])
            can_do_s2 = (u not in instance.authorizations or s2 in instance.authorizations[u])
            
            if can_do_s1 and can_do_s2:
                # ¬x[s1,u] ∨ ¬x[s2,u]
                cnf.append([-get_var(var_map, s1, u), -get_var(var_map, s2, u)])
    
    # ========================================================================
    # 约束3: Binding of Duty
    # ========================================================================
    for s1, s2 in instance.binding_duty:
        for u in range(instance.num_users):
            can_do_s1 = (u not in instance.authorizations or s1 in instance.authorizations[u])
            can_do_s2 = (u not in instance.authorizations or s2 in instance.authorizations[u])
            
            if can_do_s1 and can_do_s2:
                # x[s1,u] <-> x[s2,u]
                # (x[s1,u] -> x[s2,u]) AND (x[s2,u] -> x[s1,u])
                # (¬x[s1,u] ∨ x[s2,u]) AND (¬x[s2,u] ∨ x[s1,u])
                cnf.append([-get_var(var_map, s1, u), get_var(var_map, s2, u)])
                cnf.append([-get_var(var_map, s2, u), get_var(var_map, s1, u)])
    
    # ========================================================================
    # 约束4: At-most-k (使用Cardinality约束编码)
    # ========================================================================
    for i, (k, steps) in enumerate(instance.at_most_k):
        # 简化：如果k >= len(steps)，跳过
        if k >= len(steps):
            continue
        
        # 找出相关用户
        relevant_users = set()
        for s in steps:
            for u in range(instance.num_users):
                if u not in instance.authorizations or s in instance.authorizations[u]:
                    relevant_users.add(u)
        
        # 使用Sequential Counter编码
        # 这里简化：使用pairwise禁止k+1个用户同时使用
        relevant = list(relevant_users)
        
        # 生成所有k+1大小的组合，禁止它们同时为真
        from itertools import combinations
        for user_combo in combinations(relevant, min(k+1, len(relevant))):
            # 至少一个用户在这些步骤中未被使用
            clause = []
            for u in user_combo:
                # 创建辅助变量: u_used = OR(x[s,u] for s in steps)
                # 这里简化：直接禁止所有步骤都使用这k+1个用户
                for s in steps:
                    if u not in instance.authorizations or s in instance.authorizations[u]:
                        clause.append(-get_var(var_map, s, u))
            if clause and len(user_combo) == k + 1:
                cnf.append(clause)
    
    if progress_callback:
        progress_callback({
            'status': f'Solving SAT ({len(cnf)} clauses)...',
            'time': time.time() - start_time
        })
    
    # ========================================================================
    # 使用Glucose4求解
    # ========================================================================
    solver = Glucose4()
    
    for clause in cnf:
        solver.add_clause(clause)
    
    # 求解
    is_sat = solver.solve()
    
    result = {'sat': 'unsat', 'sol': [], 'mul_sol': '', 'exe_time': ''}
    
    if is_sat:
        result['sat'] = 'sat'
        model = solver.get_model()
        
        # 提取解
        assignment = {}
        for s in range(instance.num_steps):
            for u in range(instance.num_users):
                var_id = get_var(var_map, s, u)
                if var_id and var_id in model and model[model.index(var_id)] > 0:
                    assignment[s] = u
                    result['sol'].append(f"s{s+1}: u{u+1}")
                    break
        
        # 检查多解
        # 添加blocking clause
        blocking = [-get_var(var_map, s, u) for s, u in assignment.items()]
        solver.add_clause(blocking)
        
        if solver.solve():
            result['mul_sol'] = 'other solutions exist'
        else:
            result['mul_sol'] = 'this is the only solution'
    
    solver.delete()
    
    end_time = time.time()
    result['exe_time'] = f"{int((end_time - start_time) * 1000)}ms"
    
    if progress_callback:
        progress_callback({'status': 'Finished', 'time': end_time - start_time})
    
    return result
