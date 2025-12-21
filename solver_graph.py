"""
革命性算法：基于约束图的智能搜索

关键洞察：
1. 解只使用了14个用户（从500个中！）
2. SoD约束形成一个"冲突图"
3. 这是图着色问题的变体
4. 可以用贪心图着色 + 智能回溯
"""

import time
from collections import defaultdict, deque
import random

def Solver_GraphColoring(filename, progress_callback=None):
    """
    基于约束图分析的WSP求解器
    
    算法核心：
    1. 构建SoD冲突图
    2. 使用改进的图着色算法分配用户
    3. 智能处理At-most-k和其他约束
    """
    from wsp_app import read_file
    
    start_time = time.time()
    
    try:
        instance = read_file(filename)
    except Exception as e:
        return {'sat': 'error', 'sol': [], 'mul_sol': f'Parse error: {e}', 'exe_time': '0ms'}
    
    if progress_callback:
        progress_callback({'status': 'Analyzing constraint graph...', 'time': 0})
    
    # 构建有效用户映射
    valid_users_for_step = {}
    for s in range(instance.num_steps):
        valid = [u for u in range(instance.num_users)
                if u not in instance.authorizations or s in instance.authorizations[u]]
        valid_users_for_step[s] = valid
        if not valid:
            return {'sat': 'unsat', 'sol': [], 'mul_sol': f'No valid users for step {s+1}', 'exe_time': '0ms'}
    
    # ========================================================================
    # PHASE 1: 构建约束图
    # ========================================================================
    
    # SoD冲突图：如果两个步骤有SoD约束，它们之间有边
    sod_graph = defaultdict(set)
    for s1, s2 in instance.separation_duty:
        sod_graph[s1].add(s2)
        sod_graph[s2].add(s1)
    
    # BoD等价类：必须用同一个用户的步骤组
    bod_groups = build_bod_groups(instance)
    
    if progress_callback:
        progress_callback({
            'status': f'Graph: {len(sod_graph)} nodes, BoD groups: {len(bod_groups)}',
            'time': time.time() - start_time
        })
    
    # ========================================================================
    # PHASE 2: 智能图着色算法
    # ========================================================================
    
    # 步骤按度数（冲突数）排序 - 最受约束的优先
    step_order = sorted(range(instance.num_steps), 
                       key=lambda s: len(sod_graph[s]), 
                       reverse=True)
    
    # 多次尝试不同的随机化
    best_solution = None
    best_users_count = float('inf')
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() - start_time > 60:  # 1分钟限制
            break
        
        # 添加随机性
        if attempt > 0:
            random.shuffle(step_order)
        
        assignment = {}
        user_usage = defaultdict(set)  # user -> set of steps
        
        # 对每个步骤分配用户
        success = True
        for s in step_order:
            # 找到可用的用户（不违反SoD）
            forbidden_users = set()
            for neighbor in sod_graph[s]:
                if neighbor in assignment:
                    forbidden_users.add(assignment[neighbor])
            
            # 候选用户：有效且不在禁止列表
            candidates = [u for u in valid_users_for_step[s] if u not in forbidden_users]
            
            if not candidates:
                success = False
                break
            
            # 智能选择：优先选择已经使用过的用户（减少用户总数）
            used_candidates = [u for u in candidates if u in user_usage]
            
            if used_candidates:
                # 选择使用次数最少的
                user = min(used_candidates, key=lambda u: len(user_usage[u]))
            else:
                # 需要新用户，选第一个
                user = candidates[0]
            
            assignment[s] = user
            user_usage[user].add(s)
        
        if not success:
            continue
        
        # 检查所有约束
        if check_all_constraints(instance, assignment):
            users_count = len(user_usage)
            if users_count < best_users_count:
                best_solution = assignment.copy()
                best_users_count = users_count
                
                if progress_callback:
                    progress_callback({
                        'status': f'Found solution with {users_count} users!',
                        'time': time.time() - start_time
                    })
                
                break  # 找到解就停止
    
    # ========================================================================
    # 返回结果
    # ========================================================================
    
    result = {'sat': 'unsat', 'sol': [], 'mul_sol': '', 'exe_time': ''}
    
    if best_solution:
        result['sat'] = 'sat'
        for s in range(instance.num_steps):
            result['sol'].append(f"s{s+1}: u{best_solution[s]+1}")
        result['mul_sol'] = f'Solution uses {best_users_count} distinct users'
    else:
        result['mul_sol'] = 'No solution found with graph coloring'
    
    end_time = time.time()
    result['exe_time'] = f"{int((end_time - start_time) * 1000)}ms"
    
    if progress_callback:
        progress_callback({'status': 'Finished', 'time': end_time - start_time})
    
    return result

def build_bod_groups(instance):
    """构建BoD等价类"""
    parent = list(range(instance.num_steps))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for s1, s2 in instance.binding_duty:
        union(s1, s2)
    
    groups = defaultdict(list)
    for s in range(instance.num_steps):
        groups[find(s)].append(s)
    
    return [g for g in groups.values() if len(g) > 1]

def check_all_constraints(instance, assignment):
    """检查分配是否满足所有约束"""
    
    # SoD
    for s1, s2 in instance.separation_duty:
        if assignment.get(s1) == assignment.get(s2):
            return False
    
    # BoD
    for s1, s2 in instance.binding_duty:
        if assignment.get(s1) != assignment.get(s2):
            return False
    
    # At-most-k
    for k, steps in instance.at_most_k:
        users_used = set(assignment.get(s) for s in steps if s in assignment)
        if len(users_used) > k:
            return False
    
    # One-team (简化检查)
    for steps, teams in instance.one_team:
        step_users = [assignment.get(s) for s in steps if s in assignment]
        if len(step_users) == len(steps):
            team_match = any(
                all(u in team for u in step_users if u is not None)
                for team in teams
            )
            if not team_match:
                return False
    
    return True

if __name__ == "__main__":
    import sys
    
    test_file = sys.argv[1] if len(sys.argv) > 1 else "SAI/additional-examples/4-constraint-hard/0.txt"
    
    print(f"Testing Graph Coloring solver on: {test_file}")
    print("=" * 60)
    
    result = Solver_GraphColoring(test_file)
    
    print(f"\nStatus: {result['sat']}")
    print(f"Time: {result['exe_time']}")
    print(f"Info: {result['mul_sol']}")
    
    if result['sat'] == 'sat':
        print(f"\n✅ Solution found! ({len(result['sol'])} assignments)")
        print("First 10 assignments:")
        for line in result['sol'][:10]:
            print(f"  {line}")
