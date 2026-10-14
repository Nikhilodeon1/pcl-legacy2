import os
import logging
import pandas as pd
logging.basicConfig(level=logging.INFO)
_SEPSIS_ICD9_EXACT = {'99591', '99592', '78552'}

def _norm_icd(code):
    return str(code).upper().replace('.', '').replace(' ', '').strip()

def _is_sepsis_code(n):
    if n in _SEPSIS_ICD9_EXACT:
        return True
    if n.startswith('038'):
        return True
    if n.startswith('A40') or n.startswith('A41'):
        return True
    if n.startswith('R6520') or n.startswith('R6521'):
        return True
    return False

def _resolve(path):
    if os.path.exists(path):
        return path
    alt = path.replace('.csv.gz', '.csv')
    return alt if os.path.exists(alt) else None

def mimic_sepsis_hadm_ids(data_path):
    path = _resolve(os.path.join(data_path, 'hosp', 'diagnoses_icd.csv.gz'))
    if path is None:
        logging.warning('MIMIC hosp/diagnoses_icd.csv.gz NOT FOUND — sepsis labels will all be 0. Download it to enable ICD-based sepsis labeling.')
        return set()
    d = pd.read_csv(path, usecols=['hadm_id', 'icd_code'], dtype={'icd_code': str}, encoding_errors='replace')
    d = d[d['icd_code'].notna()]
    mask = d['icd_code'].map(lambda c: _is_sepsis_code(_norm_icd(c)))
    ids = set(d.loc[mask, 'hadm_id'].astype(int))
    logging.info(f'MIMIC sepsis labeling: {len(ids)} admissions flagged sepsis+ (ICD)')
    return ids

def eicu_sepsis_stay_ids(data_path):
    path = _resolve(os.path.join(data_path, 'diagnosis.csv.gz'))
    if path is None:
        logging.warning('eICU diagnosis.csv.gz NOT FOUND — sepsis labels will all be 0. Download it to enable ICD-based sepsis labeling.')
        return set()
    d = pd.read_csv(path, usecols=['patientunitstayid', 'icd9code'], encoding_errors='replace')
    d = d[d['icd9code'].notna()]

    def row_sepsis(code):
        return any((_is_sepsis_code(_norm_icd(part)) for part in str(code).split(',')))
    mask = d['icd9code'].map(row_sepsis)
    ids = set(d.loc[mask, 'patientunitstayid'].astype(int))
    logging.info(f'eICU sepsis labeling: {len(ids)} stays flagged sepsis+ (ICD)')
    return ids
