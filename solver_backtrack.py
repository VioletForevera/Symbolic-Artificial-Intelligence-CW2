"""
终极算法：智能回溯搜索 with 约束传播

核心思想：
1. 不是盲目搜索500个用户
2. 动态识别"候选用户集"
3. 使用约束传播大幅剪枝
4. 智能变量排序
"""

import time
from collections import defaultdict

def Solver_SmartBacktrack(filename, progress_callback=None):
    """
    智能回溯求解器 with 约束传播
    """
    from wsp_app import read_file
    
    start_time = time.time()
    
    try:
        instance = read_file(filename)
    except Exception as e:
        return {'sat': 'error', 'sol': [], 'mul_sol': f'Parse error: {e}', 'exe_time': '0ms'}
    
    if progress_callback:
        progress_callback({'status': 'Initializing backtrack search...', 'time': 0})
    
    # 构建域
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
    
    # BoD约束传播：合并域
    for s1, s2 in instance.binding_duty:
        common = domains[s1] & domains[s2]
        if not common:
            return {'sat': 'unsat', 'sol': [], 'mul_sol': 'BoD conflict', 'exe_time': '0ms'}
        domains[s1] = common
        domains[s2] = common
    
    # 构建SoD图
    sod_neighbors = defaultdict(set)
    for s1, s2 in instance.separation_duty:
        sod_neighbors[s1].add(s2)
        sod_neighbors[s2].add(s1)
    
    if progress_callback:
        progress_callback({'status': 'Starting search...', 'time': time.time() - start_time})
    
    # 智能回溯搜索
    assignment = {}
    nodes_explored = [0]
    
    def backtrack(step_idx, steps_to_assign):
        nonlocal nodes_explored
        
        # 超时检查
        if time.time() - start_time > 60:
            return False
        
        nodes_explored[0] += 1
        
        # 进度报告
        if progress_callback and nodes_explored[0] % 1000 == 0:
            progress_callback({
                'status': f'Explored {nodes_explored[0]} nodes...',
                'time': time.time() - start_time
            })
        
        # 所有步骤已分配
        if step_idx >= len(steps_to_assign):
            return check_global_constraints(instance, assignment)
        
        s = steps_to_assign[step_idx]
        
        # 约束传播：计算当前可用用户
        available = domains[s].copy()
        
        # 移除违反SoD的用户
        for neighbor in sod_neighbors[s]:
            if neighbor in assignment:
                available.discard(assignment[neighbor])
        
        if not available:
            return False  # 剪枝
        
        # 智能排序：优先尝试已使用的用户（减少总用户数）
        used_users = set(assignment.values())
        candidates = sorted(available, key=lambda u: (u not in used_users, u))
        
        # 尝试每个候选用户
        for user in candidates:
            assignment[s] = user
            
            # 递归搜索
            if backtrack(step_idx + 1, steps_to_assign):
                return True
            
            del assignment[s]
        
        return False
    
    # 智能变量排序：最受约束的步骤优先
    steps_to_assign = sorted(range(instance.num_steps),
                            key=lambda s: (len(domains[s]), -len(sod_neighbors[s])))
    
    # 执行搜索
    found = backtrack(0, steps_to_assign)
    
    result = {'sat': 'unsat', 'sol': [], 'mul_sol': '', 'exe_time': ''}
    
    if found:
        result['sat'] = 'sat'
        for s in range(instance.num_steps):
            result['sol'].append(f"s{s+1}: u{assignment[s]+1}")
        
        users_count = len(set(assignment.values()))
        result['mul_sol'] = f'Solution uses {users_count} distinct users (explored {nodes_explored[0]} nodes)'
    else:
        result['mul_sol'] = f'No solution found (explored {nodes_explored[0]} nodes)'
    
    end_time = time.time()
    result['exe_time'] = f"{int((end_time - start_time) * 1000)}ms"
    
    if progress_callback:
        progress_callback({'status': 'Finished', 'time': end_time - start_time})
    
    return result

def check_global_constraints(instance, assignment):
    """检查全局约束（At-most-k, One-team）"""
    
    # At-most-k
    for k, steps in instance.at_most_k:
        users_in_steps = set(assignment[s] for s in steps if s in assignment)
        if len(users_in_steps) > k:
            return False
    
    # One-team
    for steps, teams in instance.one_team:
        assigned_users = [assignment[s] for s in steps if s in assignment]
        if len(assigned_users) == len(steps):
            team_match = any(
                all(u in team for u in assigned_users)
                for team in teams
            )
            if not team_match:
                return False
    
    return True

if __name__ == "__main__":
    import sys
    
    test_file = sys.argv[1] if len(sys.argv) > 1 else "SAI/additional-examples/4-constraint-hard/0.txt"
    
    print(f"Testing Smart Backtrack solver on: {test_file}")
    print("=" * 60)
    
    result = Solver_SmartBacktrack(test_file)
    
    print(f"\nStatus: {result['sat']}")
    print(f"Time: {result['exe_time']}")
    print(f"Info: {result['mul_sol']}")
    
    if result['sat'] == 'sat':
        print(f"\n✅ Solution found!")
        print("First 10 assignments:")
        for line in result['sol'][:10]:
            print(f"  {line}")
