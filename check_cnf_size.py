"""
快速测试：检查SAT编码是否有问题
"""

from wsp_app import read_file
import time

filepath = "SAI/additional-examples/4-constraint-hard/0.txt"

print("读取实例...")
instance = read_file(filepath)

print(f"步骤数: {instance.num_steps}")
print(f"用户数: {instance.num_users}")
print(f"SoD约束: {len(instance.separation_duty)}")
print(f"At-most-k约束: {len(instance.at_most_k)}")

# 估算CNF大小
print("\n估算CNF公式大小...")

# 1. Assignment约束
assignment_clauses = instance.num_steps  # ALO
assignment_clauses += sum(
    len([u for u in range(instance.num_users) 
         if u not in instance.authorizations or s in instance.authorizations[u]]) ** 2 
    for s in range(instance.num_steps)
) // 2  # AMO (pairwise)

print(f"Assignment约束: ~{assignment_clauses:,} 子句")

# 2. SoD约束
sod_clauses = 0
for s1, s2 in instance.separation_duty:
    # 能同时做两个步骤的用户数
    count = sum(1 for u in range(instance.num_users)
                if (u not in instance.authorizations or s1 in instance.authorizations[u])
                and (u not in instance.authorizations or s2 in instance.authorizations[u]))
    sod_clauses += count

print(f"SoD约束: ~{sod_clauses:,} 子句")

# At-most-k (这个可能很大)
amk_clauses = 0
for k, steps in instance.at_most_k:
    if k < len(steps):
        relevant = set()
        for s in steps:
            for u in range(instance.num_users):
                if u not in instance.authorizations or s in instance.authorizations[u]:
                    relevant.add(u)
        
        # 使用组合数估算
        from math import comb
        if len(relevant) > k:
            amk_clauses += comb(len(relevant), k+1)

print(f"At-most-k约束: ~{amk_clauses:,} 子句")

total = assignment_clauses + sod_clauses + amk_clauses
print(f"\n总计约: {total:,} 子句")
print(f"变量数: {instance.num_steps * instance.num_users:,}")

if total > 1_000_000:
    print("\n⚠️  警告：CNF公式太大（>100万子句），可能需要很长时间编码！")
    print("建议：使用更高效的At-most-k编码（如Sequential Counter）")
