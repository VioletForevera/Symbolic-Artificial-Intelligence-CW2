from wsp_parser import read_file

filename = "SAI/additional-examples/4-constraint-hard/0.txt"
instance = read_file(filename)

print(f"Total Users: {instance.num_users}")
print(f"Total Steps: {instance.num_steps}")
print(f"Users with explicit authorizations: {len(instance.authorizations)}")
print(f"One-team constraints: {len(instance.one_team)}")

# Check if all users 1-500 have authorizations
all_auth_users = set(instance.authorizations.keys())
print(f"\nFirst 10 authorized users: {sorted(list(all_auth_users))[:10]}")
print(f"Last 10 authorized users: {sorted(list(all_auth_users))[-10:]}")

# Identify generic users
essential_users = set(instance.authorizations.keys())
all_users = set(range(instance.num_users))
generic_users = all_users - essential_users

print(f"\nGeneric users (no explicit auth): {len(generic_users)}")
print(f"Essential + Generic = {len(essential_users)} + {len(generic_users)} = {len(essential_users) + len(generic_users)}")
