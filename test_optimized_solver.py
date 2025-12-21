"""
测试优化后的求解器性能
Test the performance of the optimized solver
"""

import os
import time
from wsp_app import Solver, read_file

def test_single_instance(filepath):
    """测试单个实例"""
    print(f"\n{'='*70}")
    print(f"测试文件 (Testing): {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    # 读取实例信息
    try:
        instance = read_file(filepath)
        print(f"步骤数 (Steps): {instance.num_steps}")
        print(f"用户数 (Users): {instance.num_users}")
        print(f"约束数 (Constraints): {instance.num_constraints}")
        print(f"-" * 70)
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        return
    
    # 运行求解器
    result = Solver(filepath)
    
    # 显示结果
    print(f"状态 (Status): {result['sat']}")
    print(f"执行时间 (Time): {result['exe_time']}")
    
    if result['sat'] == 'sat':
        print(f"解决方案 (Solution):")
        for assignment in result['sol'][:10]:  # 只显示前10个
            print(f"  {assignment}")
        if len(result['sol']) > 10:
            print(f"  ... (共 {len(result['sol'])} 个分配)")
        print(f"多解检查 (Multiple solutions): {result.get('mul_sol', 'N/A')}")
    elif result['sat'] == 'unsat':
        print(f"原因: {result.get('mul_sol', 'No solution exists')}")
    
    print(f"{'='*70}\n")
    return result

def test_hard_instances(directory="SAI/additional-examples/4-constraint-hard", max_files=5):
    """测试困难实例"""
    print(f"\n🚀 测试困难实例 (4-constraint-hard)")
    print(f"目录: {directory}")
    print(f"最多测试: {max_files} 个文件\n")
    
    if not os.path.exists(directory):
        print(f"❌ 目录不存在: {directory}")
        return
    
    # 获取测试文件
    files = sorted([f for f in os.listdir(directory) 
                    if f.endswith('.txt') and 'solution' not in f])[:max_files]
    
    if not files:
        print(f"❌ 目录中没有找到测试文件")
        return
    
    results = []
    total_time = 0
    
    for filename in files:
        filepath = os.path.join(directory, filename)
        result = test_single_instance(filepath)
        if result:
            results.append({
                'file': filename,
                'status': result['sat'],
                'time': result['exe_time']
            })
            # 提取时间（毫秒）
            time_ms = int(result['exe_time'].replace('ms', ''))
            total_time += time_ms
    
    # 汇总报告
    print(f"\n{'='*70}")
    print(f"📊 测试汇总 (Summary)")
    print(f"{'='*70}")
    print(f"| 文件 | 状态 | 时间 |")
    print(f"|{'-'*20}|{'-'*10}|{'-'*15}|")
    for r in results:
        print(f"| {r['file']:<18} | {r['status']:<8} | {r['time']:>13} |")
    print(f"{'='*70}")
    print(f"总时间 (Total time): {total_time}ms ({total_time/1000:.2f}s)")
    print(f"平均时间 (Average): {total_time/len(results):.0f}ms")
    print(f"{'='*70}\n")

def test_basic_instances(directory="SAI/instances", start=1, end=5):
    """测试基本实例"""
    print(f"\n📝 测试基本实例 (example{start}.txt - example{end}.txt)")
    
    if not os.path.exists(directory):
        print(f"❌ 目录不存在: {directory}")
        return
    
    results = []
    
    for i in range(start, end + 1):
        filename = f"example{i}.txt"
        filepath = os.path.join(directory, filename)
        
        if os.path.exists(filepath):
            result = test_single_instance(filepath)
            if result:
                results.append({
                    'file': filename,
                    'status': result['sat'],
                    'time': result['exe_time']
                })
        else:
            print(f"⚠️  文件不存在: {filename}")
    
    # 汇总
    if results:
        print(f"\n{'='*70}")
        print(f"📊 基本实例测试汇总")
        print(f"{'='*70}")
        for r in results:
            print(f"{r['file']}: {r['status']} in {r['time']}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("🎯 WSP 优化求解器性能测试")
    print("   Optimized WSP Solver Performance Test")
    print("="*70)
    
    if len(sys.argv) > 1:
        # 测试指定文件
        test_single_instance(sys.argv[1])
    else:
        # 默认测试流程
        choice = input("\n选择测试类型:\n1. 基本实例 (Basic instances)\n2. 困难实例 (Hard instances)\n3. 两者都测试 (Both)\n\n请输入 (1/2/3) [默认: 2]: ").strip()
        
        if choice == '1':
            test_basic_instances()
        elif choice == '3':
            test_basic_instances()
            test_hard_instances()
        else:  # 默认选项 2
            test_hard_instances()
    
    print("\n✅ 测试完成！")
