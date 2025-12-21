import time
from ortools.sat.python import cp_model
from wsp_parser import read_file, Instance

def test_pruning(filename):
    """
    Test script to verify Super User Pruning is working correctly.
    Shows before/after statistics.
    """
    instance = read_file(filename)
    
    print(f"\n{'='*60}")
    print(f"Testing: {filename}")
    print(f"{'='*60}")
    print(f"Total Users in Instance: {instance.num_users}")
    print(f"Total Steps in Instance: {instance.num_steps}")
    
    # Step 1: Identify Essential Users
    essential_users = set()
    
    # Users with explicit authorizations
    for u in instance.authorizations.keys():
        essential_users.add(u)
    
    # Users mentioned in one_team constraints
    for (steps, teams) in instance.one_team:
        for team_users in teams:
            essential_users.update(team_users)
    
    print(f"Essential Users (from constraints): {len(essential_users)}")
    
    # Step 2: Identify Generic Pool
    all_users = set(range(instance.num_users))
    generic_users = all_users - essential_users
    
    print(f"Generic Users (super users): {len(generic_users)}")
    
    # Step 3: Prune
    kept_generic_users = sorted(list(generic_users))[:instance.num_steps]
    
    print(f"Kept Generic Users (pruned to {instance.num_steps} max): {len(kept_generic_users)}")
    
    # Step 4: Active Users
    active_users = essential_users | set(kept_generic_users)
    
    print(f"\n>>> PRUNING RESULT <<<")
    print(f"Active Users (will be used in model): {len(active_users)}")
    print(f"Pruned Users (discarded): {instance.num_users - len(active_users)}")
    print(f"Reduction: {100 * (instance.num_users - len(active_users)) / instance.num_users:.1f}%")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import os
    directory = "SAI/additional-examples/4-constraint-hard"
    
    # Test on the first instance
    files = sorted([f for f in os.listdir(directory) if f.endswith('.txt') and 'solution' not in f])[:1]
    
    for filename in files:
        path = os.path.join(directory, filename)
        test_pruning(path)
