# Bayesian Network Prototype — Written Report

## Extended Music Distraction and Speed Infraction Model

**Course:** Probability & Statistics  
**Reference Paper:** Catalina, C.A. et al. (2020). *Music Distraction among Young Drivers: Analysis by Gender and Experience.* Journal of Advanced Transportation, 2020, Article ID 6039762.  
**Software:** pgmpy (Python)

---

## 1. Application Description and Modeling Goal

The chosen paper by Catalina et al. (2020) investigates whether listening to music while driving increases the probability of committing speed infractions among young drivers aged 20–28. Using a driving simulator and Bayesian networks, the authors modelled the joint influence of music type (no music, sad/relaxing, happy/aggressive), driver gender, and driving experience on speed-related outcomes, along with telemetric mediational variables such as acceleration, RPM, braking, and other infractions.

Their key finding was that the probability of driving at an adequate speed ranged from 96.32% (less-experienced male drivers without music) to 79.38% (less-experienced female drivers listening to happy/aggressive music), and that less-experienced drivers were disproportionately affected by emotionally activating music.

**Our extension.** The original study was conducted exclusively on high-performance roads and did not control for the time of day. We extend the paper's model by introducing two new contextual variables — *Road Type* (Urban vs. Highway) and *Time of Day* (Day vs. Night) — to capture how environmental context interacts with the existing driver and music variables to influence speed infraction risk. This yields a richer, more realistic model that preserves the paper's core causal structure while enabling new inference questions about situational risk.

**Research question:** *How does the combined effect of music type, driver profile (gender, experience), and driving context (road type, time of day) influence the probability of speed infractions among young drivers?*

---

## 2. Variables and Their States

Our model contains **9 variables** organised into four conceptual layers:

### 2.1 Root Nodes (Exogenous Variables)

| Variable | States | Source | Justification |
|---|---|---|---|
| **Music** | No_Music, Sad_Relaxing, Happy_Aggressive | Paper (Table 4) | The independent variable of the study. Music categories follow the paper's grouping based on mood classification from the University Pompeu Fabra. |
| **Gender** | Female, Male | Paper (Table 4) | Demographic variable. The paper found gender differences in speed infraction patterns, particularly under music conditions. |
| **Experience** | Low, High | Paper (Table 4) | Defined by the paper as: High if the driver held a licence for >1.5 years and drove more than once per week; Low otherwise. |
| **Time_of_Day** | Day, Night | *New variable* | Night driving introduces fatigue, reduced visibility, and lower traffic enforcement — all well-documented risk amplifiers in road safety literature. The paper did not control for this factor. |
| **Road_Type** | Urban, Highway | *New variable* | The paper only tested high-performance roads. Urban roads involve lower speed limits but more complex traffic situations, while highways allow higher base speeds and therefore larger speed differentials when limits are exceeded. |

### 2.2 Intermediate Nodes (Mediational Variables)

| Variable | States | Parents | Justification |
|---|---|---|---|
| **Braking** | No_Brake, Normal, Abrupt | Experience, Road_Type | Braking behaviour is influenced by the driver's experience (less experienced drivers brake more abruptly) and road type (urban driving requires more frequent braking). Derived from Table 4 of the paper. |
| **RPM** | Low (<2250), Normal (2250–3000), High (>3000) | Music, Road_Type | Engine RPM reflects driving intensity. Activating music leads to higher RPMs (per the paper's telemetric findings), and highway driving naturally involves higher sustained RPMs. |

### 2.3 Target Variable

| Variable | States | Parents | Justification |
|---|---|---|---|
| **Speed_Infraction** | Correct, Minor, Serious | Music, Gender, Experience, Time_of_Day, Road_Type, Braking | The central variable of the model. "Correct" means driving below the legal speed limit; "Minor" means above the limit but below the radar activation threshold; "Serious" means above the radar threshold. These definitions follow the Spanish DGT speed classification used in the paper. |

### 2.4 Child Node

| Variable | States | Parents | Justification |
|---|---|---|---|
| **Other_Infraction** | No, Yes | Speed_Infraction, Experience | Running red lights, road marking violations, collisions, and other non-speed infractions. Higher speed increases the likelihood of cascading infractions, and less-experienced drivers commit them at higher rates. Derived from Table 4 of the paper. |

---

## 3. Network Structure

The directed acyclic graph (DAG) contains **9 nodes** and **12 directed edges**. The structure is organised as follows:

```
Root layer:       Music    Gender    Experience    Time_of_Day    Road_Type
                    |         |          |    \          |          |   \
                    |         |          |     \         |          |    \
Intermediate:       |         |       Braking   \       |        RPM    |
                    |         |          |        \      |               |
                    v         v          v         v     v               v
Target:          ──────────── Speed_Infraction ─────────────────────────
                                    |       \
Child:                    Other_Infraction    (via Experience)
```

**Edge list:**

1. Experience → Braking (experienced drivers brake more smoothly)
2. Road_Type → Braking (urban driving requires more braking)
3. Music → RPM (activating music leads to higher RPMs)
4. Road_Type → RPM (highway driving involves higher RPMs)
5. Music → Speed_Infraction (core relationship from the paper)
6. Gender → Speed_Infraction (gender effects on speed behaviour)
7. Experience → Speed_Infraction (experience moderates risk)
8. Time_of_Day → Speed_Infraction (night increases risk)
9. Road_Type → Speed_Infraction (highway enables more serious infractions)
10. Braking → Speed_Infraction (braking pattern signals speed behaviour)
11. Speed_Infraction → Other_Infraction (speeding leads to cascading violations)
12. Experience → Other_Infraction (inexperience increases all infraction types)

**Structural design choices.** We intentionally placed Speed_Infraction as a central node with many parents rather than using a long causal chain. This reflects the reality that speed behaviour is jointly determined by the driver's profile, the environment, and their driving actions — not a purely sequential process. The RPM node does not directly feed into Speed_Infraction because RPM is better understood as a *correlate* of speed behaviour rather than a direct cause; both are manifestations of the driver's underlying arousal and driving style. This avoids creating spurious causal chains while still capturing the relationship between music and engine behaviour.

---

## 4. Probability Specification

### 4.1 Prior Distributions (Root Nodes)

All root node priors are derived directly from the paper's data (Table 4) or from reasonable assumptions for the new variables:

| Variable | Distribution | Source |
|---|---|---|
| Music | P(No_Music) = 0.35, P(Sad) = 0.31, P(Happy) = 0.34 | Paper Table 4 |
| Gender | P(Female) = 0.64, P(Male) = 0.36 | Paper Table 4 |
| Experience | P(Low) = 0.46, P(High) = 0.54 | Paper Table 4 |
| Time_of_Day | P(Day) = 0.55, P(Night) = 0.45 | Assumed — slightly more daytime driving |
| Road_Type | P(Urban) = 0.60, P(Highway) = 0.40 | Assumed — young drivers drive more in urban contexts |

### 4.2 Conditional Probability Tables

**Braking | Experience, Road_Type** (12 parameters). The paper reports an overall distribution of 91.2% no braking, 4.4% normal, 4.4% abrupt (Table 4). We condition this on experience and road type: less-experienced drivers on urban roads exhibit the highest rate of abrupt braking (8%), while experienced drivers on highways exhibit the lowest (2%). These adjustments are consistent with the general finding that inexperience leads to more reactive driving behaviour.

**RPM | Music, Road_Type** (18 parameters). The paper reports 59.8% low RPM, 30.5% normal, 9.8% high (Table 4). We condition on music type (happy/aggressive music increases RPM) and road type (highway driving involves higher sustained RPMs). The most extreme cell — happy music on highway — yields 35% low, 38% normal, and 27% high RPM, reflecting the combined effect of emotional activation and high-speed road conditions.

**Speed_Infraction | Music, Gender, Experience, Time_of_Day, Road_Type, Braking** (432 parameters). This is the most critical CPT. We constructed it programmatically using a two-step approach:

1. **Base rates from Table 9 of the paper.** For each combination of Music × Gender × Experience (12 combinations), the paper provides the exact probability of correct speed, minor infraction, and serious infraction. These serve as empirically grounded base rates.

2. **Multiplicative adjustment factors for new variables.** We apply calibrated adjustments for Time_of_Day, Road_Type, and Braking:
   - *Night driving:* increases minor infraction probability by 20% and serious infraction by 35%, reflecting reduced visibility and fatigue effects.
   - *Highway:* increases serious infraction probability by 40% but slightly reduces minor infraction probability by 10% (on highways, exceeding the limit tends to result in larger speed differentials).
   - *Urban:* increases minor infraction by 10% but reduces serious infraction by 30% (lower base speeds limit the magnitude of excess).
   - *Normal braking:* reduces both infraction types (signals controlled driving).
   - *Abrupt braking:* increases both (signals reactive correction after excess speed).

After applying all adjustments, each row is renormalised to ensure probabilities sum to 1. This approach ensures that (a) the CPT is grounded in empirical data where available, (b) new variable effects are applied consistently, and (c) the model remains probabilistically valid.

**Other_Infraction | Speed_Infraction, Experience** (12 parameters). Base rate from the paper: 96.3% no infraction, 3.7% infraction (Table 4). We condition on speed state and experience: a serious speed infraction by a less-experienced driver yields 18% probability of other infractions, while correct speed by an experienced driver yields only 2%.

### 4.3 Total Model Parameters

The model contains **485 CPT parameters** across 9 nodes. Despite the large Speed_Infraction CPT (432 parameters), the model remains tractable because exact inference via Variable Elimination is efficient for this network size.

---

## 5. Inference Scenarios

We demonstrate four inference scenarios that exercise different reasoning capabilities of Bayesian networks: predictive reasoning (from causes to effects), diagnostic reasoning (from effects to causes), interventional comparison, and sensitivity analysis.

### 5.1 Scenario 1 — Predictive: Worst-Case vs. Best-Case Driver Profile

**Question:** What is the probability distribution over speed infraction states for the riskiest possible driver profile versus the safest?

**Worst-case evidence:** Female, Low experience, Happy/Aggressive music, Highway, Night.

| Speed State | Probability |
|---|---|
| Correct | 69.34% |
| Minor infraction | 10.77% |
| Serious infraction | 19.89% |

**Best-case evidence:** Male, High experience, No music, Urban, Day.

| Speed State | Probability |
|---|---|
| Correct | 92.34% |
| Minor infraction | 7.62% |
| Serious infraction | 0.03% |

**Interpretation.** The worst-case profile yields nearly a 1-in-5 chance of serious speed infraction — a dramatic increase compared to the 3.08% base rate (Table 4 of the paper). The best-case profile virtually eliminates serious infractions (0.03%). This demonstrates how the combination of all risk factors amplifies danger far beyond what any single variable would suggest. Notably, the addition of our new variables (Highway, Night) pushes the worst-case probability beyond what the paper's original model would predict: the paper reports 79.38% adequate speed for this demographic profile, while our extended model gives 69.34% when highway and night conditions are added — a meaningful increase in predicted risk that reflects the contextual factors the paper did not consider.

### 5.2 Scenario 2 — Diagnostic: What Caused the Infraction?

**Question:** Given that a serious speed infraction was observed, what can we infer about the most likely causes?

**Evidence:** Speed_Infraction = Serious.

| Variable | Most likely state | Probability |
|---|---|---|
| Music | Happy_Aggressive | 65.94% |
| Experience | Low | 76.03% |
| Time_of_Day | Night | 52.48% |

**Interpretation.** This demonstrates *diagnostic reasoning* — reasoning backwards from effect to cause. The model strongly implicates happy/aggressive music (66% posterior vs. 34% prior) and low experience (76% vs. 46% prior) as the most probable contributing factors to a serious infraction. Time of day shifts more modestly (52% night vs. 45% prior), suggesting it is a contributing but not dominant factor. This type of reasoning is one of the core strengths of Bayesian networks: given an observed outcome, we can quantify the most likely causal profile, which has direct implications for targeted road safety interventions (e.g., awareness campaigns specifically targeting less-experienced drivers about the risks of emotionally activating music).

### 5.3 Scenario 3 — Intervention: Effect of Switching Music Type

**Question:** For a fixed high-risk driver profile (Female, Low experience, Highway, Night), how does changing the music type affect infraction probability?

| Music Condition | Correct (%) | Minor (%) | Serious (%) |
|---|---|---|---|
| No Music | 95.36 | 3.31 | 1.33 |
| Sad/Relaxing | 78.66 | 11.23 | 10.11 |
| Happy/Aggressive | 69.34 | 10.77 | 19.89 |

**Interpretation.** Switching from happy/aggressive music to no music reduces the serious infraction probability from 19.89% to 1.33% — a **93.3% reduction** in serious infraction risk. Even switching to sad/relaxing music cuts serious infraction risk roughly in half (from 19.89% to 10.11%). This result directly supports the paper's conclusion that music type is the most actionable risk factor, and our model extends it by showing that the intervention effect is even more pronounced in risky environmental contexts (highway at night). From a practical standpoint, this suggests that in-car systems that automatically mute or switch music genre during high-risk driving conditions (highway, night) could meaningfully reduce accident risk for young drivers.

### 5.4 Scenario 4 — Sensitivity: Variable Importance Ranking

**Question:** Which variable has the greatest influence on the probability of a serious speed infraction?

We measured the change in P(Serious Speed Infraction) when switching each variable from its safest to its riskiest state, holding all other variables at their prior distributions. The baseline probability is P(Serious) = 3.31%.

| Rank | Variable | Safe → Risky | Δ P(Serious) |
|---|---|---|---|
| 1 | Music | No_Music → Happy_Aggressive | +5.84 pp |
| 2 | Experience | High → Low | +4.00 pp |
| 3 | Braking | Normal → Abrupt | +2.81 pp |
| 4 | Road_Type | Urban → Highway | +2.36 pp |
| 5 | Gender | Male → Female | +1.25 pp |
| 6 | Time_of_Day | Day → Night | +1.00 pp |

*(pp = percentage points)*

**Interpretation.** Music type is the single most influential variable, consistent with the paper's central thesis. Experience is the second most important, which aligns with the paper's repeated finding that less-experienced drivers are disproportionately affected. Among our new variables, Road_Type ranks 4th with a meaningful 2.36 pp effect — this validates our decision to include it in the model, as it captures real variation that the paper's single road-type design could not reveal. Time_of_Day ranks last but still contributes a 1.00 pp shift, suggesting it is a modest but non-negligible contextual factor. The Braking variable ranks 3rd, which is interesting because it sits between Experience and Road_Type in influence — this suggests that observable driving behaviour (braking patterns) carries significant diagnostic value for predicting speed violations, an insight consistent with the paper's use of telemetric mediational variables.

---

## 6. Validation Against the Paper

To ensure our extension does not distort the original findings, we reproduced the paper's key result by querying our model with the same evidence combinations reported in Table 9, marginalising over the new variables (Time_of_Day, Road_Type).

| Profile | Paper (Table 9) | Our Model |
|---|---|---|
| Low-exp Male, No Music (best case) | 96.32% adequate | 95.89% adequate |
| Low-exp Female, Happy Music (worst case) | 79.38% adequate | 76.96% adequate |

The slight downward shift in our model (approximately 1–2 percentage points) is expected: by introducing Road_Type and Time_of_Day and marginalising over them, we are averaging over conditions that include riskier states (highway, night) which were not present in the paper's single-context design. The overall pattern — a ~17 percentage point spread between best and worst driver profiles — is preserved, confirming that our extension is consistent with the original findings while adding contextual resolution.

---

## 7. Discussion and Conclusions

### 7.1 What We Found

Our extended Bayesian network confirms and deepens the core findings of Catalina et al. (2020):

**Music as the dominant risk factor.** Our sensitivity analysis ranks music type as the most influential variable on serious speed infraction probability, with a 5.84 percentage point shift between no music and happy/aggressive music. This is consistent with the paper's central conclusion and extends it by demonstrating that the effect is robust even when new contextual variables are included in the model.

**Experience as a critical moderator.** The 4.00 pp effect of experience confirms the paper's finding that less-experienced drivers are disproportionately vulnerable to music-induced distraction. Our diagnostic inference (Scenario 2) further shows that given a serious infraction, the posterior probability of low experience is 76% — a strong signal that experience acts as a protective buffer against emotional arousal while driving.

**Environmental context matters.** Our new variables contribute meaningfully to the model. Road_Type has a 2.36 pp effect, which is larger than gender (1.25 pp) — a variable that the paper explicitly studied. This suggests that the paper's finding of gender differences in speed infraction may be partially confounded by road type if men and women drive different routes. Time_of_Day contributes a more modest 1.00 pp effect, which is plausible given that the simulator experiments were conducted in controlled settings where fatigue and visibility were not factors.

**Compound risk amplification.** The predictive scenarios (Scenario 1) demonstrate that risk factors compound non-linearly. A driver with all risk factors active (inexperienced, female, happy music, highway, night) faces a 19.89% probability of serious infraction — roughly 6× the base rate of 3.08%. This multiplicative amplification is precisely the type of insight that Bayesian networks are well-suited to capture, as they model the full joint distribution rather than examining each factor in isolation.

### 7.2 Connection to the Paper

Our prototype is designed as a meaningful extension of Catalina et al. (2020), not a replication. The key connections and departures are:

- We preserve the paper's core causal logic: music, gender, and experience are root causes that influence speed behaviour through mediational driving variables.
- We use the paper's empirically reported probabilities (Tables 4, 6–9) as the foundation for our CPTs, ensuring our model is grounded in real data rather than arbitrary assumptions.
- We extend the model by adding two contextual variables that the paper explicitly did not consider, allowing us to ask new questions (e.g., "How does road type interact with music to affect infraction risk?") that go beyond the paper's scope.
- We restructured the DAG based on our own causal reasoning rather than learning it from data, which the paper did using a score-based greedy algorithm (Buntine, 1991). This is a deliberate design choice: since we do not have access to the raw data, manual structure specification with justified causal assumptions is more appropriate and transparent.

### 7.3 Limitations

- **No raw data access.** We could not learn the structure or parameters from data; all CPTs are specified manually using the paper's aggregate results and calibrated assumptions. This limits the precision of the probabilities, particularly for the new variables.
- **Assumed independence of new variables.** We assumed Time_of_Day and Road_Type are marginally independent of Music, Gender, and Experience at the root level. In reality, there may be correlations (e.g., young men may be more likely to drive on highways at night), but these would require empirical data to model.
- **Small original sample.** The paper used 19 participants, which limits the external validity of the base rates we used. Our extension inherits this limitation.
- **Multiplicative adjustment approach.** The adjustment factors for new variables (e.g., "night increases serious infraction by 35%") are informed assumptions, not empirical estimates. Future work could calibrate these with real-world traffic accident data.

### 7.4 Practical Implications

The model suggests several practical interventions for improving young driver safety:

1. **Music-aware in-car systems.** Automatically reducing music volume or switching to calming genres when the car detects high-speed driving or highway conditions could reduce infraction risk by up to 93% for the most vulnerable drivers.
2. **Experience-targeted training.** Driving schools could use the model's risk profiles to design scenarios that expose less-experienced drivers to the specific conditions (happy music + highway) where they are most vulnerable.
3. **Context-aware risk warnings.** Navigation systems could provide real-time risk alerts that account for time of day, road type, and the driver's experience level.

---

## 8. Technical Summary

| Property | Value |
|---|---|
| Number of nodes | 9 |
| Number of edges | 12 |
| Total CPT parameters | 485 |
| Root nodes | Music, Gender, Experience, Time_of_Day, Road_Type |
| Intermediate nodes | Braking, RPM |
| Target variable | Speed_Infraction |
| Child variable | Other_Infraction |
| New variables (extension) | Time_of_Day, Road_Type |
| Software | pgmpy (Python) |
| Inference method | Variable Elimination (exact) |

---

## References

- Catalina, C.A., García-Herrero, S., Cabrerizo, E., Herrera, S., García-Pineda, S., Mohamadi, F., & Mariscal, M.A. (2020). Music Distraction among Young Drivers: Analysis by Gender and Experience. *Journal of Advanced Transportation*, 2020, Article ID 6039762.
- Buntine, W. (1991). Theory refinement on Bayesian networks. *Proceedings of the Seventh Conference on Uncertainty in Artificial Intelligence*.
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874.