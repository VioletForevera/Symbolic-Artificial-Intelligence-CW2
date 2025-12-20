
import re

class Instance:
    def __init__(self, num_steps, num_users, num_constraints):
        self.num_steps = num_steps
        self.num_users = num_users
        self.num_constraints = num_constraints
        
        # User ID -> list of authorized Step IDs
        self.authorizations = {} 
        
        # List of (step1, step2) tuples
        self.separation_duty = []
        
        # List of (step1, step2) tuples
        self.binding_duty = []
        
        # List of (k, [step_ids]) tuples
        self.at_most_k = []
        
        # List of (step_ids, [team1_users, team2_users, ...]) tuples
        self.one_team = []

    def __repr__(self):
        return (f"Instance(Steps={self.num_steps}, Users={self.num_users}, "
                f"Constraints={self.num_constraints}, "
                f"Auths={len(self.authorizations)}, "
                f"SoD={len(self.separation_duty)}, "
                f"BoD={len(self.binding_duty)}, "
                f"AMK={len(self.at_most_k)}, "
                f"OneTeam={len(self.one_team)})")

def read_file(filename):
    """
    Parses the WSP input file and returns an Instance object.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    instance = None
    
    # Regex Patterns (Case Insensitive)
    header_pattern = re.compile(r'#Steps:\s*(\d+),\s*#Users:\s*(\d+),\s*#Constraints:\s*(\d+)', re.IGNORECASE)
    
    # sID and uID extraction helpers
    def parse_ids(text, prefix):
        # findall returns strings, convert to int and 0-index
        pattern = fr'{prefix}(\d+)'
        return [int(val) - 1 for val in re.findall(pattern, text, re.IGNORECASE)]

    # Line type patterns
    auth_pattern = re.compile(r'^authori[sz]ations\s+u(\d+)\s*(.*)', re.IGNORECASE)
    sod_pattern = re.compile(r'^separation-of-duty\s+s(\d+)\s+s(\d+)', re.IGNORECASE)
    bod_pattern = re.compile(r'^binding-of-duty\s+s(\d+)\s+s(\d+)', re.IGNORECASE)
    amk_pattern = re.compile(r'^at-most-k\s+(\d+)\s+(.*)', re.IGNORECASE)
    ot_pattern = re.compile(r'^one-team\s+(.*)', re.IGNORECASE)
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Parse Header if not already done
        if instance is None:
            match = header_pattern.match(line)
            if match:
                n_steps = int(match.group(1))
                n_users = int(match.group(2))
                n_const = int(match.group(3))
                instance = Instance(n_steps, n_users, n_const)
            continue
            
        # Parse Constraints
        
        # Authorizations: u[ID] s[ID] ...
        match = auth_pattern.match(line)
        if match:
            u_id = int(match.group(1)) - 1
            remainder = match.group(2)
            steps = parse_ids(remainder, 's')
            
            # If user has multiple lines, extend; otherwise set. 
            # Assuming input lines are unique per user or cumulative.
            if u_id not in instance.authorizations:
                instance.authorizations[u_id] = []
            instance.authorizations[u_id].extend(steps)
            continue
            
        # Separation-of-duty: s[ID] s[ID]
        match = sod_pattern.match(line)
        if match:
            s1 = int(match.group(1)) - 1
            s2 = int(match.group(2)) - 1
            instance.separation_duty.append((s1, s2))
            continue
            
        # Binding-of-duty: s[ID] s[ID]
        match = bod_pattern.match(line)
        if match:
            s1 = int(match.group(1)) - 1
            s2 = int(match.group(2)) - 1
            instance.binding_duty.append((s1, s2))
            continue
            
        # At-most-k: [K] s[ID] ...
        match = amk_pattern.match(line)
        if match:
            k = int(match.group(1))
            remainder = match.group(2)
            steps = parse_ids(remainder, 's')
            instance.at_most_k.append((k, steps))
            continue
            
        # One-team: s[ID]... (u[ID] ...) ...
        match = ot_pattern.match(line)
        if match:
            content = match.group(1)
            # The line structure is: steps part ... (team1) (team2) ...
            # We can split by '(' to separate the initial steps part from the team parts.
            parts = content.split('(')
            
            # First part contains steps
            steps_part = parts[0]
            steps = parse_ids(steps_part, 's')
            
            teams = []
            for team_part in parts[1:]:
                # team_part looks like "u1 u2) ..." or "u1 u2)"
                # We need to extract users before the closing ')'
                if ')' in team_part:
                    users_str = team_part.split(')')[0]
                    team_users = parse_ids(users_str, 'u')
                    teams.append(team_users)
            
            instance.one_team.append((steps, teams))
            continue
            
    return instance

if __name__ == "__main__":
    # Small test logic
    import sys
    import os
    
    # Create a dummy file if provided argument is 'test'
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_content = """#Steps: 5, #Users: 4, #Constraints: 5
Authorisations u1 s1 s2
Authorisations u2 s2 s3
Separation-of-duty s1 s2
Binding-of-duty s2 s3
At-most-k 2 s1 s2 s3
One-team s4 s5 (u1 u2) (u3 u4)
"""
        with open("test.wsp", "w") as f:
            f.write(test_content)
        
        parsed = read_file("test.wsp")
        print("Parsed Instance:", parsed)
        print("Auths:", parsed.authorizations)
        print("SoD:", parsed.separation_duty)
        print("BoD:", parsed.binding_duty)
        print("AMK:", parsed.at_most_k)
        print("OneTeam:", parsed.one_team)
        
        # Cleanup
        os.remove("test.wsp")
