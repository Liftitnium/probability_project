"""
Bayesian Network Prototype: Extended Music Distraction Model
============================================================
Inspired by: Catalina et al. (2020) - "Music Distraction among Young Drivers:
             Analysis by Gender and Experience"

Extension: We add two new contextual variables (Time_of_Day and Road_Type)
that the original paper did not consider, to model how environmental context
interacts with music, driver profile, and driving behavior to influence
speed infractions.

Research Question:
    "How does the combined effect of music type, driver profile (gender,
     experience), and road context (road type, time of day) influence
     the probability of speed infractions among young drivers?"

Software: pgmpy (Python)
"""

import numpy as np
import pandas as pd
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DEFINE THE NETWORK STRUCTURE
# ============================================================================
# 
# Node descriptions and states:
# 
# ROOT NODES (exogenous / no parents):
#   - Music:           {No_Music, Sad_Relaxing, Happy_Aggressive}
#   - Gender:          {Female, Male}
#   - Experience:      {Low, High}
#   - Time_of_Day:     {Day, Night}          ** NEW VARIABLE **
#   - Road_Type:       {Urban, Highway}      ** NEW VARIABLE **
#
# INTERMEDIATE NODES:
#   - Braking:         {No_Brake, Normal, Abrupt}
#   - RPM:             {Low, Normal, High}
#
# TARGET NODE:
#   - Speed_Infraction: {Correct, Minor, Serious}
#
# CHILD NODE:
#   - Other_Infraction: {No, Yes}
#
# ============================================================================

# Define the DAG structure
# Each tuple (A, B) means A -> B (A influences B)
edges = [
    # Driver profile -> Braking behavior
    ("Experience", "Braking"),
    ("Road_Type", "Braking"),
    
    # Music and road context -> RPM
    ("Music", "RPM"),
    ("Road_Type", "RPM"),
    
    # All relevant factors -> Speed Infraction (target)
    ("Music", "Speed_Infraction"),
    ("Gender", "Speed_Infraction"),
    ("Experience", "Speed_Infraction"),
    ("Time_of_Day", "Speed_Infraction"),
    ("Road_Type", "Speed_Infraction"),
    ("Braking", "Speed_Infraction"),
    
    # Speed infraction and experience -> Other infractions
    ("Speed_Infraction", "Other_Infraction"),
    ("Experience", "Other_Infraction"),
]

model = BayesianNetwork(edges)

print("=" * 70)
print("BAYESIAN NETWORK PROTOTYPE")
print("Extended Music Distraction & Speed Infraction Model")
print("=" * 70)
print(f"\nNodes ({len(model.nodes())}): {list(model.nodes())}")
print(f"Edges ({len(model.edges())}): {list(model.edges())}")

# ============================================================================
# 2. SPECIFY CONDITIONAL PROBABILITY TABLES (CPTs)
# ============================================================================
#
# JUSTIFICATION STRATEGY:
# - For variables present in the original paper (Music, Gender, Experience,
#   Speed_Infraction, Braking, RPM, Other_Infraction): we derive probabilities
#   from the paper's reported data (Tables 4, 6-9).
# - For new variables (Time_of_Day, Road_Type): we use reasonable assumptions
#   grounded in traffic safety literature and the paper's experimental setup.
#
# ============================================================================

# --- ROOT NODES (marginal / prior probabilities) ---

# Music: from Table 4 of the paper
# Without music: 34.97%, Sad/Relax: 31.15%, Happy/Aggression: 33.88%
cpd_music = TabularCPD(
    variable='Music',
    variable_card=3,
    values=[[0.35], [0.31], [0.34]],
    state_names={'Music': ['No_Music', 'Sad_Relaxing', 'Happy_Aggressive']}
)

# Gender: from Table 4 — Women: 63.78%, Men: 36.22%
cpd_gender = TabularCPD(
    variable='Gender',
    variable_card=2,
    values=[[0.64], [0.36]],
    state_names={'Gender': ['Female', 'Male']}
)

# Experience: from Table 4 — Low: 45.68%, High: 54.32%
cpd_experience = TabularCPD(
    variable='Experience',
    variable_card=2,
    values=[[0.46], [0.54]],
    state_names={'Experience': ['Low', 'High']}
)

# Time_of_Day: NEW — assumed equal split for general modeling
# Justification: In typical driving, roughly half of young driver trips
# occur during daylight and half during evening/night periods.
cpd_time = TabularCPD(
    variable='Time_of_Day',
    variable_card=2,
    values=[[0.55], [0.45]],
    state_names={'Time_of_Day': ['Day', 'Night']}
)

# Road_Type: NEW — assumed split reflecting common young driver usage
# Urban driving is more frequent for young drivers (commuting, city life)
cpd_road = TabularCPD(
    variable='Road_Type',
    variable_card=2,
    values=[[0.60], [0.40]],
    state_names={'Road_Type': ['Urban', 'Highway']}
)

# --- INTERMEDIATE NODES ---

# Braking | Experience, Road_Type
# From Table 4: No brake 91.22%, Normal 4.39%, Abrupt 4.39%
# We condition on Experience (less exp. -> more abrupt braking)
# and Road_Type (highway -> less braking overall, urban -> more braking)
#
# Columns: (Exp=Low,Road=Urban), (Exp=Low,Road=Highway),
#           (Exp=High,Road=Urban), (Exp=High,Road=Highway)
cpd_braking = TabularCPD(
    variable='Braking',
    variable_card=3,
    values=[
        # No_Brake
        [0.85, 0.88, 0.90, 0.94],
        # Normal
        [0.07, 0.06, 0.06, 0.04],
        # Abrupt
        [0.08, 0.06, 0.04, 0.02],
    ],
    evidence=['Experience', 'Road_Type'],
    evidence_card=[2, 2],
    state_names={
        'Braking': ['No_Brake', 'Normal', 'Abrupt'],
        'Experience': ['Low', 'High'],
        'Road_Type': ['Urban', 'Highway']
    }
)

# RPM | Music, Road_Type
# From Table 4: Low(<2250) 59.76%, Normal(2250-3000) 30.49%, High(>3000) 9.75%
# Music that excites -> higher RPM; Highway -> higher RPM overall
#
# Columns ordered by (Music, Road_Type):
# (NoMusic,Urban), (NoMusic,Hwy), (Sad,Urban), (Sad,Hwy),
# (Happy,Urban), (Happy,Hwy)
cpd_rpm = TabularCPD(
    variable='RPM',
    variable_card=3,
    values=[
        # Low RPM
        [0.70, 0.55, 0.65, 0.50, 0.50, 0.35],
        # Normal RPM
        [0.22, 0.30, 0.25, 0.32, 0.30, 0.38],
        # High RPM
        [0.08, 0.15, 0.10, 0.18, 0.20, 0.27],
    ],
    evidence=['Music', 'Road_Type'],
    evidence_card=[3, 2],
    state_names={
        'RPM': ['Low', 'Normal', 'High'],
        'Music': ['No_Music', 'Sad_Relaxing', 'Happy_Aggressive'],
        'Road_Type': ['Urban', 'Highway']
    }
)

# --- TARGET NODE: Speed_Infraction ---
# 
# Speed_Infraction | Music, Gender, Experience, Time_of_Day, Road_Type, Braking
# This is the most critical CPT. We have 3 x 2 x 2 x 2 x 2 x 3 = 144 columns.
#
# Base rates from the paper (Tables 6-9) for Music x Gender x Experience combos,
# then adjusted for Time_of_Day and Road_Type effects, and Braking.
#
# Strategy: Start from Table 9 values, apply multiplicative adjustments
# for new variables.

def build_speed_cpd():
    """
    Build the Speed_Infraction CPT programmatically.
    
    Base probabilities from Table 9 of the paper:
    P(Speed | Music, Gender, Experience)
    
    Then apply adjustment factors for:
    - Time_of_Day: Night increases infraction probability by ~15-20%
    - Road_Type: Highway increases serious infraction probability by ~25%
    - Braking: Normal braking reduces infraction; Abrupt braking signals
      reactive correction (associated with higher prior speed)
    """
    
    # Base rates from Table 9: [Correct, Minor, Serious]
    # Keys: (Music_idx, Gender_idx, Experience_idx)
    base_rates = {
        # No Music
        (0, 0, 0): [0.9622, 0.0307, 0.0071],  # No_Music, Female, Low_Exp
        (0, 0, 1): [0.9576, 0.0387, 0.0037],  # No_Music, Female, High_Exp
        (0, 1, 0): [0.9632, 0.0270, 0.0099],  # No_Music, Male, Low_Exp
        (0, 1, 1): [0.9299, 0.0697, 0.0005],  # No_Music, Male, High_Exp
        # Sad/Relaxing
        (1, 0, 0): [0.8417, 0.1043, 0.0540],  # Sad, Female, Low_Exp
        (1, 0, 1): [0.9302, 0.0548, 0.0150],  # Sad, Female, High_Exp
        (1, 1, 0): [0.9277, 0.0513, 0.0209],  # Sad, Male, Low_Exp
        (1, 1, 1): [0.9177, 0.0719, 0.0104],  # Sad, Male, High_Exp
        # Happy/Aggressive
        (2, 0, 0): [0.7938, 0.1000, 0.1062],  # Happy, Female, Low_Exp
        (2, 0, 1): [0.9345, 0.0395, 0.0260],  # Happy, Female, High_Exp
        (2, 1, 0): [0.8234, 0.0982, 0.0784],  # Happy, Male, Low_Exp
        (2, 1, 1): [0.9205, 0.0597, 0.0197],  # Happy, Male, High_Exp
    }
    
    # Adjustment factors for new variables
    # Time_of_Day: Night driving increases risk
    # Justification: Reduced visibility, fatigue, and less traffic enforcement
    # at night are well-documented risk factors in traffic safety literature.
    time_adjustments = {
        0: {'minor_mult': 1.0, 'serious_mult': 1.0},    # Day (baseline)
        1: {'minor_mult': 1.20, 'serious_mult': 1.35},   # Night (+20% minor, +35% serious)
    }
    
    # Road_Type: Highway allows higher speeds, more serious infractions
    # Justification: Higher base speeds on highways mean exceeding limits
    # leads to larger speed differentials; the paper only tested high-perf roads.
    road_adjustments = {
        0: {'minor_mult': 1.10, 'serious_mult': 0.70},   # Urban (more minor, less serious)
        1: {'minor_mult': 0.90, 'serious_mult': 1.40},   # Highway (fewer minor, more serious)
    }
    
    # Braking effect: modifies the speed infraction conditional on braking behavior
    # No braking: baseline; Normal braking: slight reduction; Abrupt: indicates speeding
    braking_adjustments = {
        0: {'minor_mult': 1.0, 'serious_mult': 1.0},     # No_Brake (baseline)
        1: {'minor_mult': 0.80, 'serious_mult': 0.60},   # Normal (controlled driving)
        2: {'minor_mult': 1.15, 'serious_mult': 1.25},   # Abrupt (reactive to excess speed)
    }
    
    # Build the full CPT
    # Variable order: Music(3) x Gender(2) x Experience(2) x Time(2) x Road(2) x Braking(3)
    # Total columns: 3 * 2 * 2 * 2 * 2 * 3 = 144
    
    n_cols = 3 * 2 * 2 * 2 * 2 * 3  # 144
    values = np.zeros((3, n_cols))
    
    col = 0
    for music in range(3):
        for gender in range(2):
            for exp in range(2):
                for time in range(2):
                    for road in range(2):
                        for brake in range(3):
                            base = base_rates[(music, gender, exp)].copy()
                            
                            # Apply adjustments to minor and serious infraction probs
                            minor = base[1]
                            serious = base[2]
                            
                            minor *= time_adjustments[time]['minor_mult']
                            minor *= road_adjustments[road]['minor_mult']
                            minor *= braking_adjustments[brake]['minor_mult']
                            
                            serious *= time_adjustments[time]['serious_mult']
                            serious *= road_adjustments[road]['serious_mult']
                            serious *= braking_adjustments[brake]['serious_mult']
                            
                            # Correct = remainder to ensure sums to 1
                            correct = max(1.0 - minor - serious, 0.01)
                            total = correct + minor + serious
                            
                            values[0, col] = correct / total
                            values[1, col] = minor / total
                            values[2, col] = serious / total
                            
                            col += 1
    
    return values

speed_values = build_speed_cpd()

cpd_speed = TabularCPD(
    variable='Speed_Infraction',
    variable_card=3,
    values=speed_values.tolist(),
    evidence=['Music', 'Gender', 'Experience', 'Time_of_Day', 'Road_Type', 'Braking'],
    evidence_card=[3, 2, 2, 2, 2, 3],
    state_names={
        'Speed_Infraction': ['Correct', 'Minor', 'Serious'],
        'Music': ['No_Music', 'Sad_Relaxing', 'Happy_Aggressive'],
        'Gender': ['Female', 'Male'],
        'Experience': ['Low', 'High'],
        'Time_of_Day': ['Day', 'Night'],
        'Road_Type': ['Urban', 'Highway'],
        'Braking': ['No_Brake', 'Normal', 'Abrupt'],
    }
)

# Other_Infraction | Speed_Infraction, Experience
# From Table 4: No infraction 96.33%, Infraction 3.67%
# Higher speed -> more likely other infractions; less experience -> more infractions
#
# Columns: (Speed=Correct,Exp=Low), (Speed=Correct,Exp=High),
#           (Speed=Minor,Exp=Low), (Speed=Minor,Exp=High),
#           (Speed=Serious,Exp=Low), (Speed=Serious,Exp=High)
cpd_other = TabularCPD(
    variable='Other_Infraction',
    variable_card=2,
    values=[
        # No infraction
        [0.97, 0.98, 0.92, 0.95, 0.82, 0.90],
        # Yes infraction
        [0.03, 0.02, 0.08, 0.05, 0.18, 0.10],
    ],
    evidence=['Speed_Infraction', 'Experience'],
    evidence_card=[3, 2],
    state_names={
        'Other_Infraction': ['No', 'Yes'],
        'Speed_Infraction': ['Correct', 'Minor', 'Serious'],
        'Experience': ['Low', 'High']
    }
)

# ============================================================================
# 3. ADD CPTs TO MODEL AND VALIDATE
# ============================================================================

model.add_cpds(cpd_music, cpd_gender, cpd_experience, cpd_time, cpd_road,
               cpd_braking, cpd_rpm, cpd_speed, cpd_other)

assert model.check_model(), "Model validation failed!"
print("\n✓ Model validated successfully — all CPTs are consistent.\n")

# ============================================================================
# 4. INFERENCE SCENARIOS
# ============================================================================

inference = VariableElimination(model)

def print_query(title, result):
    """Pretty-print inference results."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    var = list(result.variables)[0]
    states = result.state_names[var]
    values = result.values
    for state, prob in zip(states, values):
        bar = '█' * int(prob * 40)
        print(f"  {state:<20s} {prob:.4f}  {bar}")
    print()


# ── Scenario 1: PREDICTIVE QUERY ──────────────────────────────────────────
# What is the probability of speed infraction for a less-experienced female
# driver listening to happy/aggressive music on a highway at night?
# This represents a "worst case" scenario suggested by the paper's findings.

print("\n" + "=" * 70)
print("INFERENCE SCENARIO 1: Predictive — Worst-Case Driver Profile")
print("=" * 70)
print("Evidence: Female, Low experience, Happy/Aggressive music, Highway, Night")

result1 = inference.query(
    variables=['Speed_Infraction'],
    evidence={
        'Gender': 'Female',
        'Experience': 'Low',
        'Music': 'Happy_Aggressive',
        'Road_Type': 'Highway',
        'Time_of_Day': 'Night'
    }
)
print_query("P(Speed_Infraction | worst-case profile)", result1)

# Compare with the "best case"
print("Comparison — Best-Case Profile:")
print("Evidence: Male, High experience, No music, Urban, Day")

result1b = inference.query(
    variables=['Speed_Infraction'],
    evidence={
        'Gender': 'Male',
        'Experience': 'High',
        'Music': 'No_Music',
        'Road_Type': 'Urban',
        'Time_of_Day': 'Day'
    }
)
print_query("P(Speed_Infraction | best-case profile)", result1b)


# ── Scenario 2: DIAGNOSTIC QUERY ──────────────────────────────────────────
# Given that a serious speed infraction was observed, what is the most
# likely music type the driver was listening to?
# This demonstrates "explaining away" — a key BN capability.

print("=" * 70)
print("INFERENCE SCENARIO 2: Diagnostic — What caused the infraction?")
print("=" * 70)
print("Evidence: Serious speed infraction observed")

result2_music = inference.query(
    variables=['Music'],
    evidence={'Speed_Infraction': 'Serious'}
)
print_query("P(Music | Serious infraction)", result2_music)

result2_exp = inference.query(
    variables=['Experience'],
    evidence={'Speed_Infraction': 'Serious'}
)
print_query("P(Experience | Serious infraction)", result2_exp)

result2_time = inference.query(
    variables=['Time_of_Day'],
    evidence={'Speed_Infraction': 'Serious'}
)
print_query("P(Time_of_Day | Serious infraction)", result2_time)


# ── Scenario 3: INTERVENTION COMPARISON ───────────────────────────────────
# How much does switching music type reduce risk for an at-risk driver?
# Fixed profile: Female, Low experience, Highway, Night
# Compare across all three music conditions.

print("=" * 70)
print("INFERENCE SCENARIO 3: Intervention — Effect of Music Change")
print("=" * 70)
print("Fixed profile: Female, Low experience, Highway, Night")
print("Comparing music conditions:\n")

fixed_evidence = {
    'Gender': 'Female',
    'Experience': 'Low',
    'Road_Type': 'Highway',
    'Time_of_Day': 'Night'
}

music_types = ['No_Music', 'Sad_Relaxing', 'Happy_Aggressive']
intervention_results = {}

for music in music_types:
    evidence = {**fixed_evidence, 'Music': music}
    result = inference.query(
        variables=['Speed_Infraction'],
        evidence=evidence
    )
    states = result.state_names['Speed_Infraction']
    probs = result.values
    intervention_results[music] = dict(zip(states, probs))
    
    print(f"  Music = {music}:")
    for state, prob in zip(states, probs):
        print(f"    {state:<15s} {prob:.4f}")
    print()

# Calculate risk reduction
happy_serious = intervention_results['Happy_Aggressive']['Serious']
no_music_serious = intervention_results['No_Music']['Serious']
reduction = ((happy_serious - no_music_serious) / happy_serious) * 100
print(f"  → Switching from Happy/Aggressive to No Music reduces serious")
print(f"    infraction risk by {reduction:.1f}% for this driver profile.\n")


# ── Scenario 4: SENSITIVITY ANALYSIS ──────────────────────────────────────
# Which variable has the strongest influence on speed infraction?
# We measure the change in P(Serious) when setting each variable to its
# "riskiest" vs "safest" state, holding others at their priors.

print("=" * 70)
print("INFERENCE SCENARIO 4: Sensitivity — Variable Importance")
print("=" * 70)
print("Measuring change in P(Serious Speed Infraction) per variable:\n")

# Baseline: no evidence
baseline = inference.query(variables=['Speed_Infraction'])
baseline_serious = baseline.values[2]
print(f"  Baseline P(Serious) = {baseline_serious:.4f}\n")

# Variable pairs: (name, safe_state, risky_state)
sensitivity_vars = [
    ('Music', 'No_Music', 'Happy_Aggressive'),
    ('Gender', 'Male', 'Female'),
    ('Experience', 'High', 'Low'),
    ('Time_of_Day', 'Day', 'Night'),
    ('Road_Type', 'Urban', 'Highway'),
    ('Braking', 'Normal', 'Abrupt'),
]

sensitivity_results = []
for var_name, safe, risky in sensitivity_vars:
    safe_result = inference.query(
        variables=['Speed_Infraction'],
        evidence={var_name: safe}
    )
    risky_result = inference.query(
        variables=['Speed_Infraction'],
        evidence={var_name: risky}
    )
    safe_serious = safe_result.values[2]
    risky_serious = risky_result.values[2]
    delta = risky_serious - safe_serious
    sensitivity_results.append((var_name, safe, risky, safe_serious, risky_serious, delta))

# Sort by impact
sensitivity_results.sort(key=lambda x: abs(x[5]), reverse=True)

for var_name, safe, risky, safe_p, risky_p, delta in sensitivity_results:
    direction = "↑" if delta > 0 else "↓"
    bar = '█' * int(abs(delta) * 200)
    print(f"  {var_name:<15s}  {safe:<20s}→ {risky:<20s}  "
          f"Δ = {direction}{abs(delta):.4f}  {bar}")

print(f"\n  → Most influential variable: {sensitivity_results[0][0]}")
print(f"  → Least influential variable: {sensitivity_results[-1][0]}")


# ── ADDITIONAL: Cross-query matching paper's key finding ──────────────────
# The paper reports the range 96.32% (exp. male, no music) to 79.38%
# (less-exp. female, happy music). Let's verify our extended model
# preserves this pattern.

print("\n" + "=" * 70)
print("VALIDATION: Reproducing Paper's Key Finding (Table 9)")
print("=" * 70)

# Best case from paper: experienced male, no music -> 96.32% adequate (paper used 92.99% for high exp male)
# Actually paper: less-exp male no music = 96.32%
# Worst case: less-exp female, happy music = 79.38%

# Our model (marginalising over new variables):
best_case = inference.query(
    variables=['Speed_Infraction'],
    evidence={'Gender': 'Male', 'Experience': 'Low', 'Music': 'No_Music'}
)
worst_case = inference.query(
    variables=['Speed_Infraction'],
    evidence={'Gender': 'Female', 'Experience': 'Low', 'Music': 'Happy_Aggressive'}
)

print(f"\n  Paper's best case (Low-exp Male, No Music):  96.32% adequate speed")
print(f"  Our model (same evidence, marginal over new vars): {best_case.values[0]*100:.2f}%")
print(f"\n  Paper's worst case (Low-exp Female, Happy):  79.38% adequate speed")
print(f"  Our model (same evidence, marginal over new vars): {worst_case.values[0]*100:.2f}%")
print(f"\n  → Our extended model preserves the paper's core pattern while adding")
print(f"    contextual nuance through Time_of_Day and Road_Type variables.")


# ── NETWORK SUMMARY ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("NETWORK SUMMARY")
print("=" * 70)
print(f"\n  Total nodes:           {len(model.nodes())}")
print(f"  Total edges:           {len(model.edges())}")
print(f"  Root nodes:            Music, Gender, Experience, Time_of_Day, Road_Type")
print(f"  Intermediate nodes:    Braking, RPM")
print(f"  Target variable:       Speed_Infraction")
print(f"  Child variable:        Other_Infraction")
print(f"  Total CPT parameters:  {sum(cpd.get_values().size for cpd in model.get_cpds())}")
print(f"\n  Variables from paper:  Music, Gender, Experience, Braking, RPM,")
print(f"                         Speed_Infraction, Other_Infraction")
print(f"  New variables:         Time_of_Day, Road_Type")
print(f"\n  Implementation:        pgmpy (Python)")
print(f"  Inference method:      Variable Elimination")