import os
import logging
import numpy as np
import pandas as pd
from .variables import CANONICAL_VARIABLES, EICU_VITAL_MAPPING, EICU_VITAL_FALLBACK, EICU_LAB_MAPPING, VAR_TO_IDX, PLAUS
from .preprocessing import HOURS, MIN_LOS_H, preprocess_timeseries, has_hemodynamic_window, MinMaxNormalizer, make_heartbeat
from .sofa_sepsis import eicu_sofa_sepsis_labels
logging.basicConfig(level=logging.INFO)
_PERIODIC_COLS = {col: var for var, col in EICU_VITAL_MAPPING.items()}
_APERIODIC_COLS = {col: var for var, col in EICU_VITAL_FALLBACK.items()}
_LAB_NAME_MAP = {name.lower(): var for var, name in EICU_LAB_MAPPING.items()}
_EICU_CHUNK = 1000000
_MAX_MIN = HOURS * 60
_MIN_OFFSET = -60

def _compact(df):
    if df.empty:
        return df
    out = df.groupby(['patientunitstayid', 'hour', 'var'], as_index=False)['valuenum'].median()
    out['patientunitstayid'] = out['patientunitstayid'].astype('int32')
    out['hour'] = out['hour'].astype('int16')
    out['valuenum'] = out['valuenum'].astype('float32')
    out['var'] = out['var'].astype('category')
    return out

def _resolve_path(path):
    if os.path.exists(path):
        return path
    alt = path.replace('.csv.gz', '.csv')
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(f'Cannot find {path} or {alt}')

def _try_read(path):
    return pd.read_csv(_resolve_path(path), low_memory=False, encoding_errors='replace')

def _read_vitals_chunked(path, col_map, stay_ids, offset_col='observationoffset'):
    resolved = _resolve_path(path)
    header = pd.read_csv(resolved, nrows=0, encoding_errors='replace').columns
    present = [c for c in col_map if c in header]
    if not present:
        return pd.DataFrame(columns=['patientunitstayid', 'hour', 'var', 'valuenum'])
    usecols = ['patientunitstayid', offset_col] + present
    parts = []
    scanned = n_chunks = 0
    beat = make_heartbeat(f'read {os.path.basename(path)}')
    for chunk in pd.read_csv(resolved, usecols=usecols, chunksize=_EICU_CHUNK, encoding_errors='replace'):
        n_chunks += 1
        scanned += len(chunk)
        beat(n_chunks, extra=f'{scanned:,} rows scanned')
        chunk = chunk[chunk['patientunitstayid'].isin(stay_ids)]
        chunk = chunk[(chunk[offset_col] > _MIN_OFFSET) & (chunk[offset_col] < _MAX_MIN)]
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk['hour'] = (chunk[offset_col] / 60).astype(int)
        melted = chunk.melt(id_vars=['patientunitstayid', 'hour'], value_vars=present, var_name='eicu_col', value_name='valuenum')
        melted = melted[melted['valuenum'].notna()]
        melted['var'] = melted['eicu_col'].map(col_map)
        parts.append(_compact(melted[['patientunitstayid', 'hour', 'var', 'valuenum']]))
    if not parts:
        return pd.DataFrame(columns=['patientunitstayid', 'hour', 'var', 'valuenum'])
    return pd.concat(parts, ignore_index=True)

def _read_labs_chunked(path, stay_ids):
    resolved = _resolve_path(path)
    usecols = ['patientunitstayid', 'labresultoffset', 'labresult', 'labname']
    parts = []
    scanned = n_chunks = 0
    beat = make_heartbeat(f'read {os.path.basename(path)}')
    for chunk in pd.read_csv(resolved, usecols=usecols, chunksize=_EICU_CHUNK, encoding_errors='replace'):
        n_chunks += 1
        scanned += len(chunk)
        beat(n_chunks, extra=f'{scanned:,} rows scanned')
        chunk = chunk[chunk['patientunitstayid'].isin(stay_ids)]
        chunk = chunk[chunk['labresultoffset'].notna() & chunk['labresult'].notna()]
        chunk = chunk[(chunk['labresultoffset'] > _MIN_OFFSET) & (chunk['labresultoffset'] < _MAX_MIN)]
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk['hour'] = (chunk['labresultoffset'] / 60).astype(int)
        chunk['var'] = chunk['labname'].str.strip().str.lower().map(_LAB_NAME_MAP)
        chunk = chunk[chunk['var'].notna()]
        parts.append(_compact(chunk[['patientunitstayid', 'hour', 'var', 'labresult']].rename(columns={'labresult': 'valuenum'})))
    if not parts:
        return pd.DataFrame(columns=['patientunitstayid', 'hour', 'var', 'valuenum'])
    return pd.concat(parts, ignore_index=True)

def _read_nursecharting_chunked(path, stay_ids):
    try:
        resolved = _resolve_path(path)
    except FileNotFoundError:
        logging.warning('eICU nurseCharting not found — Temp will stay sparse')
        return pd.DataFrame(columns=['patientunitstayid', 'hour', 'var', 'valuenum'])
    wanted = {'temperature (c)': 'Temp', 'respiratory rate': 'Resp'}
    usecols = ['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevalname', 'nursingchartvalue']
    parts = []
    scanned = n_chunks = 0
    beat = make_heartbeat(f'read {os.path.basename(path)}')
    for chunk in pd.read_csv(resolved, usecols=usecols, chunksize=_EICU_CHUNK, encoding_errors='replace', low_memory=False):
        n_chunks += 1
        scanned += len(chunk)
        beat(n_chunks, extra=f'{scanned:,} rows scanned')
        chunk = chunk[chunk['patientunitstayid'].isin(stay_ids)]
        chunk = chunk[(chunk['nursingchartoffset'] > _MIN_OFFSET) & (chunk['nursingchartoffset'] < _MAX_MIN)]
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk['var'] = chunk['nursingchartcelltypevalname'].astype(str).str.strip().str.lower().map(wanted)
        chunk = chunk[chunk['var'].notna()]
        if chunk.empty:
            continue
        chunk['valuenum'] = pd.to_numeric(chunk['nursingchartvalue'], errors='coerce')
        chunk = chunk[chunk['valuenum'].notna()]
        chunk['hour'] = (chunk['nursingchartoffset'] / 60).astype(int)
        parts.append(_compact(chunk[['patientunitstayid', 'hour', 'var', 'valuenum']]))
    if not parts:
        return pd.DataFrame(columns=['patientunitstayid', 'hour', 'var', 'valuenum'])
    return pd.concat(parts, ignore_index=True)

def _build_all_timeseries(long_df, stay_ids, pid_to_idx):
    n_stays = len(stay_ids)
    n_vars = len(CANONICAL_VARIABLES)
    all_ts = np.full((n_stays, HOURS, n_vars), np.nan, dtype=np.float32)
    if long_df.empty:
        return all_ts
    ev = long_df[long_df['patientunitstayid'].isin(stay_ids)].copy()
    ev = ev[(ev['hour'] >= 0) & (ev['hour'] < HOURS)]
    ev['col_idx'] = ev['var'].map(VAR_TO_IDX)
    ev = ev[ev['col_idx'].notna()]
    ev['col_idx'] = ev['col_idx'].astype(int)
    for var, (lo, hi) in PLAUS.items():
        mask = ev['var'] == var
        ev.loc[mask, 'valuenum'] = ev.loc[mask, 'valuenum'].clip(lo, hi)
    ev['_arr_idx'] = ev['patientunitstayid'].map(pid_to_idx)
    ev = ev[ev['_arr_idx'].notna()]
    ev['_arr_idx'] = ev['_arr_idx'].astype(int)
    agg = ev.groupby(['_arr_idx', 'hour', 'col_idx'])['valuenum'].median().reset_index()
    all_ts[agg['_arr_idx'].values, agg['hour'].values, agg['col_idx'].values] = agg['valuenum'].values
    return all_ts

def load_eicu(data_path, fraction=1.0, seed=42, keep_raw=False):
    patients = _try_read(os.path.join(data_path, 'patient.csv.gz'))
    patients = patients[patients['age'].notna()]
    patients['age_numeric'] = pd.to_numeric(patients['age'].replace('> 89', 90), errors='coerce')
    patients = patients[patients['age_numeric'] >= 18]
    patients['los_h'] = patients['unitdischargeoffset'] / 60.0
    patients = patients[patients['los_h'] >= MIN_LOS_H].reset_index(drop=True)
    if fraction < 1.0:
        n_keep = max(1, int(len(patients) * fraction))
        patients = patients.sample(n=n_keep, random_state=seed).reset_index(drop=True)
    stay_ids = set(patients['patientunitstayid'].values)
    pid_to_idx = {pid: i for i, pid in enumerate(patients['patientunitstayid'].values)}
    sofa_labels, _ = eicu_sofa_sepsis_labels(data_path, patients, mode='single')
    logging.info(f'eICU: loading {len(stay_ids)} stays')
    logging.info('eICU: reading vitalPeriodic (chunked)...')
    long_p = _read_vitals_chunked(os.path.join(data_path, 'vitalPeriodic.csv.gz'), _PERIODIC_COLS, stay_ids)
    logging.info('eICU: reading vitalAperiodic (chunked)...')
    long_a = _read_vitals_chunked(os.path.join(data_path, 'vitalAperiodic.csv.gz'), _APERIODIC_COLS, stay_ids)
    logging.info('eICU: reading lab (chunked)...')
    long_labs = _read_labs_chunked(os.path.join(data_path, 'lab.csv.gz'), stay_ids)
    logging.info('eICU: reading nurseCharting (chunked)...')
    long_nc = _read_nursecharting_chunked(os.path.join(data_path, 'nurseCharting.csv.gz'), stay_ids)
    all_ts = _build_all_timeseries(long_p, stay_ids, pid_to_idx)
    del long_p
    for _frame in ('long_a', 'long_labs', 'long_nc'):
        _src_df = locals()[_frame]
        _src = _build_all_timeseries(_src_df, stay_ids, pid_to_idx)
        fill = np.isnan(all_ts) & ~np.isnan(_src)
        all_ts[fill] = _src[fill]
        del _src, _src_df
    del long_a, long_labs, long_nc
    import gc
    gc.collect()
    all_raw_ts = []
    all_samples = []
    skipped = {'empty': 0, 'no_hemo': 0}
    beat = make_heartbeat('eICU assembly', total=len(patients))
    for k, (_, pat) in enumerate(patients.iterrows()):
        beat(k)
        pid = pat['patientunitstayid']
        raw_ts = all_ts[pid_to_idx[pid]]
        if np.all(np.isnan(raw_ts)):
            skipped['empty'] += 1
            continue
        if not has_hemodynamic_window(raw_ts):
            skipped['no_hemo'] += 1
            continue
        mortality = int(pat.get('unitdischargestatus', '') == 'Expired')
        mortality_hospital = int(pat.get('hospitaldischargestatus', '') == 'Expired')
        los_3d = int(pat['los_h'] > 72)
        los_h_val = float(pat['los_h'])
        sepsis = int(sofa_labels.get(int(pid), 0))
        hospital_id = pat.get('hospitalid', 0)
        unit_type = str(pat.get('unittype', 'eICU_Unknown'))
        all_raw_ts.append(raw_ts)
        all_samples.append({'raw_ts': raw_ts, 'site_id': 3, 'patient_id': str(pid), 'mortality': mortality, 'mortality_hospital': mortality_hospital, 'los_3d': los_3d, 'los_h': los_h_val, 'sepsis': sepsis, 'hospital_id': hospital_id, 'unit_type': unit_type})
    logging.info(f'eICU: {len(all_samples)} stays loaded, skipped: {skipped}')
    normalizer = MinMaxNormalizer()
    normalizer.fit(all_raw_ts)
    processed = []
    beat = make_heartbeat('eICU preprocess', total=len(all_samples))
    for j, sample in enumerate(all_samples):
        beat(j)
        result = preprocess_timeseries(sample['raw_ts'], normalizer)
        entry = {'values': result['values'], 'mask': result['mask'], 'abg_mask': result['abg_mask'], 'c_mask': result['c_mask'], 'label': np.full(HOURS, float(sample['sepsis'])), 'site_id': sample['site_id'], 'patient_id': sample['patient_id'], 'mortality': sample['mortality'], 'mortality_hospital': sample['mortality_hospital'], 'los_3d': sample['los_3d'], 'los_h': sample['los_h'], 'hospital_id': sample.get('hospital_id', 0), 'unit_type': sample.get('unit_type', 'eICU_Unknown')}
        if keep_raw:
            entry['raw_ts'] = sample['raw_ts']
        processed.append(entry)
    return (processed, normalizer)
