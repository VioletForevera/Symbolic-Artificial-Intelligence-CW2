"""
Z3-based WSP Solver
使用微软Z3 SMT求解器 - 工业级约束求解器

Z3的优势：
- 强大的布尔约束和理论结合
- 高效的冲突驱动学习
- 支持整数约束
- 免费且广泛使用
"""

import time
from z3 import *

def Solver_Z3(filename, progress_callback=None):
    """
    使用Z3 SMT求解器解决WSP问题
    
    编码方式：使用整数变量而不是布尔变量
    - p[s] = 分配给步骤s的用户ID
    - 域约束：p[s] ∈ valid_users_for_step[s]
    """
    from wsp_app import read_file
    
    start_time = time.time()
    
    # 解析实例
    try:
        instance = read_file(filename)
    except Exception as e:
        return {'sat': 'error', 'sol': [], 'mul_sol': f'Parse error: {e}', 'exe_time': '0ms'}
    
    if progress_callback:
        progress_callback({'status': 'Building Z3 model...', 'time': 0})
    
    # 构建有效用户映射
    valid_users_for_step = {}
    for s in range(instance.num_steps):
        valid = []
        for u in range(instance.num_users):
            if u in instance.authorizations:
                if s in instance.authorizations[u]:
                    valid.append(u)
            else:
                valid.append(u)  # 超级用户
        valid_users_for_step[s] = valid
        
        if not valid:
            return {
                'sat': 'unsat',
                'sol': [],
                'mul_sol': f'Step {s+1} has no valid users',
                'exe_time': '0ms'
            }
    
    # 创建Z3求解器
    solver = Solver()
    
    # 设置超时（以毫秒为单位）
    solver.set("timeout", 180000)  # 3分钟
    
    # 创建整数变量：p[s] = 分配给步骤s的用户
    p = {}
    for s in range(instance.num_steps):
        p[s] = Int(f'p_{s}')
        
        # 域约束：p[s] 必须在有效用户中
        valid_us = valid_users_for_step[s]
        solver.add(Or([p[s] == u for u in valid_us]))
    
    # ========================================================================
    # 约束1: Separation of Duty
    # ========================================================================
    for s1, s2 in instance.separation_duty:
        solver.add(p[s1] != p[s2])
    
    # ========================================================================
    # 约束2: Binding of Duty
    # ========================================================================
    for s1, s2 in instance.binding_duty:
        solver.add(p[s1] == p[s2])
    
    # ========================================================================
    # 约束3: At-most-k
    # ========================================================================
    for i, (k, steps) in enumerate(instance.at_most_k):
        if k >= len(steps):
            continue  # 平凡满足
        
        # 找出相关用户
        involved_users = set()
        for s in steps:
            involved_users.update(valid_users_for_step[s])
        
        # 对每个用户，创建布尔变量表示是否使用
        for_each_user = []
        for u in involved_users:
            # u_used = 用户u在这些步骤中被使用
            u_used = Bool(f'amk_{i}_u{u}')
            
            # u_used <-> (p[s1]==u OR p[s2]==u OR ...)
            usage_in_steps = [p[s] == u for s in steps if u in valid_users_for_step[s]]
            
            if usage_in_steps:
                solver.add(u_used == Or(usage_in_steps))
                for_each_user.append(u_used)
        
        # 最多k个用户被使用
        if for_each_user:
            solver.add(AtMost(*for_each_user, k))
    
    # ========================================================================
    # 约束4: One-team
    # ========================================================================
    for i, (steps, teams) in enumerate(instance.one_team):
        viable_team_vars = []
        
        for t_idx, team_users in enumerate(teams):
            # 检查团队可行性
            is_viable = all(
                any(u in valid_users_for_step[s] for u in team_users if u < instance.num_users)
                for s in steps
            )
            
            if is_viable:
                team_var = Bool(f'team_{i}_{t_idx}')
                viable_team_vars.append(team_var)
                
                # 如果选择此团队，所有步骤必须由团队成员完成
                for s in steps:
                    team_members = [u for u in team_users 
                                   if u < instance.num_users and u in valid_users_for_step[s]]
                    if team_members:
                        solver.add(Implies(team_var, Or([p[s] == u for u in team_members])))
        
        if not viable_team_vars:
            return {
                'sat': 'unsat',
                'sol': [],
                'mul_sol': 'No viable team',
                'exe_time': '0ms'
            }
        
        # 恰好选择一个团队
        solver.add(Or(viable_team_vars))
        solver.add(AtMost(*viable_team_vars, 1))
    
    if progress_callback:
        progress_callback({
            'status': 'Solving with Z3...',
            'assertions': len(solver.assertions()),
            'time': time.time() - start_time
        })
    
    # ========================================================================
    # 求解
    # ========================================================================
    check_result = solver.check()
    
    result = {'sat': 'unsat', 'sol': [], 'mul_sol': '', 'exe_time': ''}
    
    if check_result == sat:
        result['sat'] = 'sat'
        model = solver.model()
        
        # 提取解
        assignment = {}
        for s in range(instance.num_steps):
            user_val = model[p[s]].as_long()
            assignment[s] = user_val
            result['sol'].append(f"s{s+1}: u{user_val+1}")
        
        # 检查多解
        # 添加blocking clause
        blocking = Or([p[s] != assignment[s] for s in range(instance.num_steps)])
        solver.add(blocking)
        
        check_result2 = solver.check()
        if check_result2 == sat:
            result['mul_sol'] = 'other solutions exist'
        else:
            result['mul_sol'] = 'this is the only solution'
    
    elif check_result == unsat:
        result['mul_sol'] = 'Problem is infeasible (Z3 proved UNSAT)'
    else:  # unknown
        result['mul_sol'] = 'Z3 timeout or unknown'
    
    end_time = time.time()
    result['exe_time'] = f"{int((end_time - start_time) * 1000)}ms"
    
    if progress_callback:
        progress_callback({'status': 'Finished', 'time': end_time - start_time})
    
    return result

if __name__ == "__main__":
    import sys
    
    test_file = sys.argv[1] if len(sys.argv) > 1 else "SAI/additional-examples/4-constraint-hard/0.txt"
    
    print(f"Testing Z3 solver on: {test_file}")
    print("=" * 60)
    
    result = Solver_Z3(test_file)
    
    print(f"\nStatus: {result['sat']}")
    print(f"Time: {result['exe_time']}")
    print(f"Info: {result['mul_sol']}")
    
    if result['sat'] == 'sat':
        print(f"\nSolution found! ({len(result['sol'])} assignments)")
        print("First 10 assignments:")
        for line in result['sol'][:10]:
            print(f"  {line}")
