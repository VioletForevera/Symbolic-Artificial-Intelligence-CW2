"""
诊断脚本：分析 WSP 实例的复杂度和瓶颈
"""

import time
from wsp_app import read_file

def diagnose_instance(filepath):
    """诊断单个实例"""
    print(f"\n{'='*80}")
    print(f"诊断文件: {filepath}")
    print(f"{'='*80}\n")
    
    start = time.time()
    instance = read_file(filepath)
    
    print(f"📊 基本信息:")
    print(f"  步骤数: {instance.num_steps}")
    print(f"  用户数: {instance.num_users}")
    print(f"  约束数: {instance.num_constraints}")
    
    # 分析用户类型
    essential_users = set(instance.authorizations.keys())
    for steps, teams in instance.one_team:
        for team_users in teams:
            essential_users.update(u for u in team_users if u < instance.num_users)
    
    all_users = set(range(instance.num_users))
    generic_users = all_users - essential_users
    
    print(f"\n👥 用户分析:")
    print(f"  关键用户 (Essential): {len(essential_users)}")
    print(f"  超级用户 (Generic): {len(generic_users)}")
    print(f"  剪枝后活跃用户: {len(essential_users) + min(len(generic_users), instance.num_steps)}")
    
    # 分析每个步骤的有效用户数
    active_users = list(essential_users) + sorted(list(generic_users))[:instance.num_steps]
    
    valid_users_for_step = {}
    for s in range(instance.num_steps):
        valid = []
        for u in active_users:
            if u in instance.authorizations:
                if s in instance.authorizations[u]:
                    valid.append(u)
            else:
                valid.append(u)
        valid_users_for_step[s] = valid
    
    # 找出最受约束的步骤
    step_constraints = [(s, len(valid_users_for_step[s])) for s in range(instance.num_steps)]
    step_constraints.sort(key=lambda x: x[1])
    
    print(f"\n🔒 步骤约束分析:")
    print(f"  最少选择的步骤: s{step_constraints[0][0]+1} (只有 {step_constraints[0][1]} 个候选用户)")
    print(f"  平均每步候选用户: {sum(len(v) for v in valid_users_for_step.values()) / instance.num_steps:.1f}")
    
    if step_constraints[0][1] == 0:
        print(f"  ⚠️  警告: 步骤 s{step_constraints[0][0]+1} 没有有效用户!")
    
    # 分析约束复杂度
    print(f"\n⚙️  约束分析:")
    print(f"  SoD (分离职责): {len(instance.separation_duty)}")
    print(f"  BoD (绑定职责): {len(instance.binding_duty)}")
    print(f"  At-most-k: {len(instance.at_most_k)}")
    print(f"  One-team: {len(instance.one_team)}")
    
    # 分析冲突
    print(f"\n🔍 冲突检测:")
    
    # BoD 冲突
    bod_conflicts = 0
    for s1, s2 in instance.binding_duty:
        common = set(valid_users_for_step[s1]) & set(valid_users_for_step[s2])
        if not common:
            bod_conflicts += 1
            print(f"  ❌ BoD 冲突: s{s1+1} 和 s{s2+1} 没有共同用户")
    
    if bod_conflicts == 0:
        print(f"  ✅ 没有 BoD 冲突")
    
    # One-team 可行性
    if instance.one_team:
        print(f"\n👥 One-team 约束详情:")
        for i, (steps, teams) in enumerate(instance.one_team):
            print(f"  约束 {i+1}: {len(steps)} 个步骤, {len(teams)} 个团队")
            
            viable_count = 0
            for t_idx, team_users in enumerate(teams):
                is_viable = True
                for s in steps:
                    if not any(u in valid_users_for_step[s] for u in team_users if u < instance.num_users):
                        is_viable = False
                        break
                if is_viable:
                    viable_count += 1
            
            print(f"    可行团队: {viable_count}/{len(teams)}")
            if viable_count == 0:
                print(f"    ❌ 警告: 没有可行团队!")
    
    # At-most-k 分析
    if instance.at_most_k:
        print(f"\n🔢 At-most-k 约束详情:")
        for i, (k, steps) in enumerate(instance.at_most_k):
            involved = set()
            for s in steps:
                involved.update(valid_users_for_step[s])
            
            print(f"  约束 {i+1}: 最多 {k} 个用户在 {len(steps)} 个步骤中")
            print(f"    潜在涉及用户: {len(involved)}")
            
            if k < len(steps):
                # 检查是否有 SoD 冲突导致需要更多用户
                min_needed = 1
                for s1 in steps:
                    for s2 in steps:
                        if s1 < s2 and (s1, s2) in instance.separation_duty:
                            min_needed = max(min_needed, 2)
                
                if k < min_needed:
                    print(f"    ❌ 可能不可满足: k={k} 但至少需要 {min_needed} 个用户")
    
    elapsed = time.time() - start
    print(f"\n⏱️  诊断耗时: {elapsed*1000:.0f}ms")
    print(f"{'='*80}\n")
    
    return {
        'num_steps': instance.num_steps,
        'num_users': instance.num_users,
        'essential_users': len(essential_users),
        'generic_users': len(generic_users),
        'active_users': len(essential_users) + min(len(generic_users), instance.num_steps),
        'min_candidates': step_constraints[0][1],
        'bod_conflicts': bod_conflicts
    }

if __name__ == "__main__":
    import os
    import sys
    
    if len(sys.argv) > 1:
        # 诊断指定文件
        diagnose_instance(sys.argv[1])
    else:
        # 诊断 hard 实例
        directory = "SAI/additional-examples/4-constraint-hard"
        if os.path.exists(directory):
            print("🔍 诊断 4-constraint-hard 实例...\n")
            files = sorted([f for f in os.listdir(directory) 
                          if f.endswith('.txt') and 'solution' not in f])[:3]
            
            for filename in files:
                filepath = os.path.join(directory, filename)
                diagnose_instance(filepath)
        else:
            print(f"❌ 目录不存在: {directory}")
            print("使用方法: python diagnose_wsp.py <文件路径>")
