import json

# 1. Load Rules at Startup
with open('osha_rules.json', 'r') as f:
    OSHA_RULES = json.load(f)

def check_violation(detected_objects):
    """
    detected_objects: List of strings from YOLO-World 
    e.g. ['bare_hand', 'industrial_machine', 'helmet']
    """
    violations_found = []

    for rule in OSHA_RULES:
        # Check if ANY trigger is present
        trigger_hit = any(t in detected_objects for t in rule['triggers'])
        
        # Check if ANY required context is present (or if none is required)
        if not rule['required_context']:
            context_hit = True
        else:
            context_hit = any(c in detected_objects for c in rule['required_context'])

        # If we have a Trigger + Context, it's a violation
        if trigger_hit and context_hit:
            violations_found.append({
                "code": rule['code'],
                "title": rule['title'],
                "text": rule['legal_text'],
                "penalty": rule['penalty_max']
            })

    return violations_found