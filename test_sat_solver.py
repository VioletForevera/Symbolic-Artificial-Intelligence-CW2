"""
测试新的SAT求解器 vs CP-SAT求解器
"""

import time
from solver_sat import Solver_SAT
from wsp_app import Solver as Solver_CPSAT

def test_both_solvers(filepath):
    print(f"\n{'='*80}")
    print(f"测试文件: {filepath}")
    print(f"{'='*80}\n")
    
    # 测试SAT求解器
    print("🔵 方法1: 专业SAT求解器 (Glucose4)")
    print("-" * 40)
    start = time.time()
    try:
        result_sat = Solver_SAT(filepath)
        elapsed_sat = time.time() - start
        
        print(f"状态: {result_sat['sat']}")
        print(f"时间: {result_sat['exe_time']}")
        print(f"多解: {result_sat['mul_sol']}")
        if result_sat['sat'] == 'sat':
            print(f"找到解！共{len(result_sat['sol'])}个分配")
    except Exception as e:
        print(f"❌ 错误: {e}")
        elapsed_sat = time.time() - start
    
    print()
    
    # 测试CP-SAT求解器
    print("🟢 方法2: Google CP-SAT求解器")
    print("-" * 40)
    start = time.time()
    try:
        result_cpsat = Solver_CPSAT(filepath)
        elapsed_cpsat = time.time() - start
        
        print(f"状态: {result_cpsat['sat']}")
        print(f"时间: {result_cpsat['exe_time']}")
        print(f"多解: {result_cpsat['mul_sol']}")
        if result_cpsat['sat'] == 'sat':
            print(f"找到解！共{len(result_cpsat['sol'])}个分配")
    except Exception as e:
        print(f"❌ 错误: {e}")
        elapsed_cpsat = time.time() - start
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_both_solvers(sys.argv[1])
    else:
        # 测试困难实例
        test_file = "SAI/additional-examples/4-constraint-hard/0.txt"
        test_both_solvers(test_file)
