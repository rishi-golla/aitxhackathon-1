# Video Generation Prompts for OSHA Violations

Use these prompts with AI video generators (Runway, Pika, Sora, etc.) to create demo footage showing clear OSHA violations that our system can detect.

---

## Current Detection Classes (YOLO-World)

```python
["bare_hand", "gloved_hand", "safety_glasses", "face", "industrial_machine"]
```

## Current OSHA Rules Triggers

| Rule | Triggers | Context Required |
|------|----------|------------------|
| Hand Protection (1910.138a) | `bare_hand`, `exposed_skin` | `industrial_machine`, `sharp_tool`, `chemical_container`, `spinning_blade` |
| Machine Guarding (1910.212a1) | `unguarded_belt`, `exposed_gears`, `rotating_parts`, `flying_sparks` | None |
| Point of Operation (1910.212a3ii) | `hand_near_blade`, `hand_in_machine_zone` | `active_machine` |

---

## VIDEO PROMPTS

### 1. BARE HANDS + INDUSTRIAL MACHINE (Primary Violation)

**Prompt:**
> "First-person POV footage of a factory worker operating a metal grinding machine. The worker's bare hands are clearly visible in frame, no gloves. Sparks flying from the grinder. Industrial workshop setting with metal shelving in background. Steady camera, realistic lighting. 10 seconds."

**Expected Detection:**
- `bare_hand` (trigger) + `industrial_machine` (context)
- **Violation: 29 CFR 1910.138(a)** - $16,131

---

### 2. NO SAFETY GLASSES + SPARKS

**Prompt:**
> "First-person POV of a worker using an angle grinder on metal pipe. Camera shows worker's face reflected in metal surface - no safety glasses visible. Bright sparks flying toward camera. Warehouse/factory setting. 8 seconds."

**Expected Detection:**
- `face` visible + no `safety_glasses` + sparks
- **Violation: 29 CFR 1910.133(a)(1)** - Eye Protection

---

### 3. HAND NEAR SPINNING BLADE

**Prompt:**
> "Close-up POV footage of a table saw or circular saw in operation. A bare hand reaches toward the spinning blade to push material through. The hand comes dangerously close to the point of operation. Industrial woodshop setting. 6 seconds."

**Expected Detection:**
- `bare_hand` + `industrial_machine` (spinning blade context)
- **Violation: 29 CFR 1910.212(a)(3)(ii)** - Point of Operation

---

### 4. UNGUARDED ROTATING MACHINERY

**Prompt:**
> "POV footage walking through a factory floor. Camera pans across machinery with exposed belts, pulleys, and rotating shafts. No guards or covers visible on the equipment. Worker's hand briefly enters frame pointing at the machinery. 12 seconds."

**Expected Detection:**
- `rotating_parts`, `exposed_gears` visible
- **Violation: 29 CFR 1910.212(a)(1)** - Machine Guarding

---

### 5. CORRECT PPE (Negative Example)

**Prompt:**
> "First-person POV of a factory worker properly equipped. Gloved hands visible operating controls. Safety glasses reflection visible. Operating a CNC machine or drill press with all guards in place. Clean, well-lit industrial facility. 8 seconds."

**Expected Detection:**
- `gloved_hand` + `safety_glasses` + `industrial_machine`
- **No violation** - Proper PPE in use

---

### 6. CHEMICAL HANDLING WITHOUT GLOVES

**Prompt:**
> "POV footage of a worker in a chemical storage area. Bare hands visible picking up and moving containers with hazard labels. Yellow/red chemical drums visible. Industrial cleaning or manufacturing setting. 10 seconds."

**Expected Detection:**
- `bare_hand` + `chemical_container` context
- **Violation: 29 CFR 1910.138(a)** - Hand Protection

---

### 7. ASSEMBLY LINE CLOSE CALL

**Prompt:**
> "First-person footage on a factory assembly line. Worker's bare hands assembling components. Suddenly reaches near a moving conveyor belt with pinch points. Quick reflexive pull-back motion. Realistic factory ambient sound. 8 seconds."

**Expected Detection:**
- `bare_hand` + `industrial_machine`
- **Violation: 29 CFR 1910.212(a)(3)(ii)** - Point of Operation

---

## PROMPT TIPS FOR BETTER DETECTION

### DO:
- Use **first-person/POV perspective** (matches Egocentric-10K dataset style)
- Show **hands clearly in frame** (main detection target)
- Include **industrial equipment** in background
- Use **realistic factory lighting** (not too dark)
- Keep **camera relatively stable** (reduces motion blur)
- Duration: **6-15 seconds** per clip

### DON'T:
- Avoid extreme close-ups (need context for violation)
- Avoid dark/shadowy scenes (detection needs visibility)
- Avoid fast camera movements (causes blur)
- Don't show cartoonish/unrealistic scenarios

---

## BATCH GENERATION TEMPLATE

For generating multiple variations quickly:

```
Base prompt: "First-person POV industrial factory footage. [WORKER_ACTION]. [PPE_STATUS]. [MACHINE_TYPE] visible. Realistic lighting, 10 seconds."

Variables:
- WORKER_ACTION: "operating grinder", "handling chemicals", "near conveyor belt", "using table saw"
- PPE_STATUS: "bare hands visible, no gloves", "wearing safety glasses", "no eye protection"
- MACHINE_TYPE: "metal grinding machine", "CNC lathe", "hydraulic press", "conveyor system"
```

---

## FILE NAMING CONVENTION

Save generated videos as:
```
violation_[rule_code]_[description]_v[version].mp4

Examples:
violation_1910138a_bare_hands_grinder_v1.mp4
violation_1910212a_unguarded_belt_v1.mp4
compliant_proper_ppe_assembly_v1.mp4
```

Place files in `/data/` folder for automatic detection by the backend.

---

## QUICK TEST CHECKLIST

After generating videos:

1. [ ] Place MP4 in `data/` folder
2. [ ] Restart backend: `python server/main.py`
3. [ ] Check console for "Found X video files"
4. [ ] Open `/dashboard/live` - verify stream appears
5. [ ] Watch for red bounding boxes on bare hands
6. [ ] Check `/status` endpoint for violation JSON
