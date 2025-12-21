"""
WSP求解器选择器
根据问题规模自动选择最优算法
"""

def Solver_Smart(filename, progress_callback=None):
    """
    智能求解器：根据问题特征自动选择算法
    
    - 小实例 (<100用户): CP-SAT
    - 中等实例 (100-300用户): CP-SAT with optimizations  
    - 大实例 (>300用户): 尝试SAT，如果失败fallback到CP-SAT
    """
    from wsp_app import read_file, Solver as Solver_CPSAT
    import time
    
    start_time = time.time()
    
    try:
        instance = read_file(filename)
    except Exception as e:
        return {'sat': 'error', 'sol': [], 'mul_sol': f'Parse error: {e}', 'exe_time': '0ms'}
    
    # 根据规模选择策略
    if instance.num_users < 100:
        # 小实例：直接用CP-SAT
        return Solver_CPSAT(filename, progress_callback)
    
    elif instance.num_users >= 400:
        # 大实例：优先尝试SAT求解器
        if progress_callback:
            progress_callback({'status': 'Trying SAT solver...', 'time': 0})
        
        try:
            from solver_sat import Solver_SAT
            
            # 给SAT求解器120秒
            result = Solver_SAT(filename, progress_callback)
            
            elapsed = time.time() - start_time
            
            # 如果SAT求解器在合理时间内找到解，返回
            if result['sat'] == 'sat' and elapsed < 120:
                return result
            
            # 否则fallback到CP-SAT
            if progress_callback:
                progress_callback({'status': 'SAT timeout, trying CP-SAT...', 'time': elapsed})
            
        except Exception as e:
            # SAT求解器失败，fallback
            if progress_callback:
                progress_callback({'status': f'SAT error: {e}, using CP-SAT...', 'time': time.time() - start_time})
        
        # Fallback到CP-SAT
        return Solver_CPSAT(filename, progress_callback)
    
    else:
        # 中等实例：CP-SAT
        return Solver_CPSAT(filename, progress_callback)

# 替换wsp_app.py中的Solver函数
def get_replacement_code():
    return """
# 在wsp_app.py中，导入智能求解器
from solver_smart import Solver_Smart

# 将原来的Solver重命名为Solver_CPSAT
# 然后创建新的Solver作为wrapper
def Solver(filename, progress_callback=None):
    return Solver_Smart(filename, progress_callback)
"""

if __name__ == "__main__":
    print("智能求解器模块已加载")
    print(get_replacement_code())
