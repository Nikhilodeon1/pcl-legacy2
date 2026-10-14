import os
CORE_VARIABLES = ['HR', 'SBP', 'DBP', 'MAP', 'SpO2', 'pH', 'HCO3', 'pCO2', 'PaO2']
EXPANDED_VARIABLES = ['Temp', 'Resp', 'WBC', 'Lactate', 'BUN', 'Creatinine', 'Potassium', 'Glucose']
USE_EXPANDED = os.environ.get('PCL_EXPANDED_VARS', '1') == '1'
CANONICAL_VARIABLES = CORE_VARIABLES + (EXPANDED_VARIABLES if USE_EXPANDED else [])
VAR_TO_IDX = {v: i for i, v in enumerate(CANONICAL_VARIABLES)}
PLAUS = {'HR': (20, 300), 'SBP': (40, 300), 'DBP': (20, 200), 'MAP': (20, 200), 'SpO2': (50, 100), 'pH': (6.5, 7.9), 'HCO3': (5, 60), 'pCO2': (10, 150), 'PaO2': (20, 700), 'Temp': (25.0, 43.0), 'Resp': (4, 60), 'WBC': (0.1, 100), 'Lactate': (0.1, 30), 'BUN': (1, 200), 'Creatinine': (0.1, 20), 'Potassium': (1.5, 9.0), 'Glucose': (20, 1000)}
ABG_VARIABLES = ['pH', 'HCO3', 'pCO2', 'PaO2']
HEMO_VARIABLES = ['SBP', 'DBP', 'MAP']
PHYSIONET_MAPPING = {'HR': 'HR', 'SBP': 'SBP', 'DBP': 'DBP', 'MAP': 'MAP', 'SpO2': 'O2Sat', 'pH': 'pH', 'HCO3': 'HCO3', 'pCO2': 'PaCO2'}
if USE_EXPANDED:
    PHYSIONET_MAPPING.update({'Temp': 'Temp', 'Resp': 'Resp', 'WBC': 'WBC', 'Lactate': 'Lactate', 'BUN': 'BUN', 'Creatinine': 'Creatinine', 'Potassium': 'Potassium', 'Glucose': 'Glucose'})
PHYSIONET_EXTRA = {'SaO2': 'SaO2'}
MIMIC_ITEM_IDS = {'SBP': [220179, 220050], 'DBP': [220180, 220051], 'MAP': [220181, 220052], 'HR': [220045], 'SpO2': [220277], 'pH': [50820], 'HCO3': [50882], 'pCO2': [50818], 'PaO2': [50821]}
if USE_EXPANDED:
    MIMIC_ITEM_IDS.update({'Temp': [223762, 223761], 'Resp': [220210, 224690], 'WBC': [51301], 'Lactate': [50813], 'BUN': [51006], 'Creatinine': [50912], 'Potassium': [50971], 'Glucose': [50931]})
MIMIC_EXPANDED_CHART = ['Temp', 'Resp']
MIMIC_EXPANDED_LAB = ['WBC', 'Lactate', 'BUN', 'Creatinine', 'Potassium', 'Glucose']
MIMIC_UNIT_CONVERT = {223761: lambda v: (v - 32.0) * 5.0 / 9.0}
EICU_VITAL_MAPPING = {'HR': 'heartrate', 'SBP': 'systemicsystolic', 'DBP': 'systemicdiastolic', 'MAP': 'systemicmean', 'SpO2': 'sao2'}
EICU_VITAL_FALLBACK = {'SBP': 'noninvasivesystolic', 'DBP': 'noninvasivediastolic', 'MAP': 'noninvasivemean'}
EICU_LAB_MAPPING = {'pH': 'pH', 'HCO3': 'bicarbonate', 'pCO2': 'paCO2', 'PaO2': 'paO2'}
if USE_EXPANDED:
    EICU_LAB_MAPPING.update({'WBC': 'WBC x 1000', 'Lactate': 'lactate', 'BUN': 'BUN', 'Creatinine': 'creatinine', 'Potassium': 'potassium', 'Glucose': 'glucose'})
    EICU_VITAL_MAPPING.update({'Temp': 'temperature', 'Resp': 'respiration'})
