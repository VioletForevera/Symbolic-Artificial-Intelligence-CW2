"""
全新算法：贪心+回溯启发式求解器
适用于高约束密度的大规模WSP实例

核心思路：
1. 贪心构造初始解
2. 识别违反的约束
3. 局部调整修复
4. 回溯搜索
"""

import random
import time
from collections import defaultdict

def Solver_Heuristic(filename, progress_callback=None, max_time=180):
    """
    启发式WSP求解器
    """
    from wsp_app import read_file
    
    start_time = time.time()
    
    try:
        instance = read_file(filename)
    except Exception as e:
        return {'sat': 'error', 'sol': [], 'mul_sol': f'Parse error: {e}', 'exe_time': '0ms'}
    
    if progress_callback:
        progress_callback({'status': 'Heuristic search...', 'time': 0})
    
    # 构建有效用户列表
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
    
    # 贪心初始化
    assignment = greedy_assign(instance, valid_users_for_step)
    
    # 迭代改进
    best_violations = count_violations(instance, assignment)
    best_assignment = assignment.copy()
    
    iterations = 0
    max_iterations = 10000
    no_improve_count = 0
    
    while time.time() - start_time < max_time and iterations < max_iterations:
        iterations += 1
        
        if best_violations == 0:
            # 找到解！
            break
        
        # 局部搜索
        new_assignment = local_search_step(instance, assignment, valid_users_for_step)
        new_violations = count_violations(instance, new_assignment)
        
        # 接受改进的解
        if new_violations < best_violations:
            best_violations = new_violations
            best_assignment = new_assignment.copy()
            assignment = new_assignment
            no_improve_count = 0
            
            if progress_callback and iterations % 100 == 0:
                progress_callback({
                    'status': f'Violations: {best_violations}',
                    'time': time.time() - start_time
                })
        else:
            no_improve_count += 1
            
            # 模拟退火：有概率接受较差的解
            if random.random() < 0.1:
                assignment = new_assignment
        
        # 如果长时间没改进，重启
        if no_improve_count > 500:
            assignment = greedy_assign(instance, valid_users_for_step)
            no_improve_count = 0
    
    result = {'sat': 'unsat', 'sol': [], 'mul_sol': '', 'exe_time': ''}
    
    if best_violations == 0:
        result['sat'] = 'sat'
        for s in range(instance.num_steps):
            result['sol'].append(f"s{s+1}: u{best_assignment[s]+1}")
        result['mul_sol'] = 'unknown (heuristic method)'
    else:
        result['mul_sol'] = f'No solution found ({best_violations} violations remaining)'
    
    end_time = time.time()
    result['exe_time'] = f"{int ((end_time - start_time) * 1000)}ms"
    
    if progress_callback:
        progress_callback({'status': 'Finished', 'time': end_time - start_time})
    
    return result

def greedy_assign(instance, valid_users):
    """贪心构造初始解"""
    assignment = {}
    used_counts = defaultdict(int)
    
    # 按约束最多的步骤优先分配
    step_order = sorted(range(instance.num_steps), 
                       key=lambda s: len(valid_users[s]))
    
    for s in step_order:
        # 选择使用次数最少的有效用户
        candidates = valid_users[s]
        if candidates:
            user = min(candidates, key=lambda u: used_counts[u])
            assignment[s] = user
            used_counts[user] += 1
        else:
            assignment[s] = 0  # Fallback
    
    return assignment

def count_violations(instance, assignment):
    """计算违反的约束数量"""
    violations = 0
    
    # SoD violations
    for s1, s2 in instance.separation_duty:
        if assignment[s1] == assignment[s2]:
            violations += 1
    
    # BoD violations  
    for s1, s2 in instance.binding_duty:
        if assignment[s1] != assignment[s2]:
            violations += 1
    
    # At-most-k violations
    for k, steps in instance.at_most_k:
        users_used = len(set(assignment[s] for s in steps))
        if users_used > k:
            violations += (users_used - k)
    
    # One-team violations (简化)
    for steps, teams in instance.one_team:
        assigned_users = [assignment[s] for s in steps]
        team_match = any(
            all(assignment[s] in team for s in steps)
            for team in teams
        )
        if not team_match:
            violations += 1
    
    return violations

def local_search_step(instance, assignment, valid_users):
    """局部搜索一步"""
    new_assignment = assignment.copy()
    
    # 随机选择一个步骤重新分配
    s = random.randint(0, instance.num_steps - 1)
    candidates = valid_users[s]
    
    if candidates:
        # 尝试换成不同的用户
        current_user = assignment[s]
        other_users = [u for u in candidates if u != current_user]
        if other_users:
            new_assignment[s] = random.choice(other_users)
    
    return new_assignment

if __name__ == "__main__":
    import sys
    test_file = sys.argv[1] if len(sys.argv) > 1 else "SAI/additional-examples/4-constraint-hard/0.txt"
    
    result = Solver_Heuristic(test_file, max_time=60)
    print(f"Status: {result['sat']}")
    print(f"Time: {result['exe_time']}")
    print(f"Info: {result['mul_sol']}")
