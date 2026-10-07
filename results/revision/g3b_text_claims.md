# G3b qualitative claims audit

| id | claim | verdict | detail |
|---|---|---|---|
| C1 | repr. dist. / MMD are larger at the far targets (MIMIC, eICU) than at near-domain PhysioNet-B ('more distant databases in representation space') | **FALSE** | mean repr_dist PN-B/MIMIC/eICU = 6.96/3.74/5.18; mean MMD = 0.294/0.100/0.184 -> both criteria rank PN-B as the MOST distant target |
| C2.repr_dist | pooled rho(repr_dist)=+0.896 has the opposite sign from every within-target value | **FALSE** | within-target physionet_b:+0.33, mimic:+0.63, eicu:+0.33; same sign as pooled at: ['physionet_b', 'mimic', 'eicu'] |
| C2.mmd | pooled rho(mmd)=+0.888 has the opposite sign from every within-target value | **FALSE** | within-target physionet_b:-0.30, mimic:+0.35, eicu:+0.02; same sign as pooled at: ['mimic', 'eicu'] |
| C3.los | PCL never wins a single LOS cell (target x seed, 9 cells) | **TRUE** | PCL best in: [] |
| C3.mort | PCL never wins a single mortality cell (6 cells) | **TRUE** | PCL best in: [] |
| C4 | PCL is the lowest-scoring method in the majority of LOS cells | **TRUE** | PCL lowest in 8/9 cells: [('physionet_b', 42), ('physionet_b', 43), ('mimic', 42), ('mimic', 43), ('mimic', 44), ('eicu', 42), ('eicu', 43), ('eicu', 44)] |
| C5 | 'same DRO >= ERM > PCL ranking' replicates on LOS | **FALSE (DRO>=ERM holds at 0/3 LOS targets)** | LOS physionet_b: erm > dro > pcl (erm=0.2286, pcl=0.2081, dro=0.2246) | LOS mimic: erm > dro > pcl (erm=-0.1223, pcl=-0.1340, dro=-0.1284) | LOS eicu: erm > dro > pcl (erm=-0.1030, pcl=-0.1143, dro=-0.1096) | MORT mimic->eicu: dro > erm > pcl (erm=0.7600, pcl=0.7440, dro=0.7640) | MORT eicu->mimic: erm > dro > pcl (erm=0.8192, pcl=0.8083, dro=0.8191) |
| C6 | violation and recon. error are correctly signed (negative rho, lower-is-better) at every LOS target | **TRUE** | physionet_b:-0.72/-0.73; mimic:-0.32/-0.55; eicu:-0.45/-0.60 (n=9 checkpoints per target; permutation p in a4_selection_los.csv: significant only at physionet_b) |
