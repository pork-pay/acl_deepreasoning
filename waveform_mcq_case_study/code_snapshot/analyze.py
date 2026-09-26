#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG 分析：从 12 导联信号 + fiducial 计算关键参数（口径与 PTB-XL-Plus 对齐）。

口径（与交接文档 / unig 对齐）：
  HR         = 60000 / median(RR_ms)
  QRS dur    = median(qrs_off - qrs_on)   ms
  PR         = median(qrs_on - p_on)      ms
  QT         = median(t_off - qrs_on)     ms
  QTc(Bazett)= QT / sqrt(RR_ms/1000)
  ST_Amp80ms = signal[J+80ms] - baseline  (mV, 逐导联)
  QRS 电轴   = atan2(  II净振幅 ,   aVF近似? )  —— 这里用 I/III 代数和（I+III 法，与 FeatureDB 一致）
"""
import math
import numpy as np
from load_fiducials import detect_rpeaks, delineate_lead

LIMB = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF']
CHEST = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
ORDER = LIMB + CHEST


def _col(sig_name, lead):
    return [s.upper() for s in sig_name].index(lead.upper())


def axis_from_i_iii(amp_i, amp_iii):
    """I/III 代数和（R-Q-S 净面积近似为正向振幅）估电轴度数。"""
    try:
        deg = math.degrees(math.atan2(amp_iii, amp_i))
        if deg <= -90: deg += 360
        return round(deg, 1)
    except Exception:
        return None


def analyze(sig, fields, fs, rpeaks=None, verbose=False):
    """sig:(N,12) mV, fields=wfdb fields(dict 含 sig_name)。返回分析 dict。"""
    sig_name = fields['sig_name']
    if rpeaks is None:
        rpeaks = detect_rpeaks(sig, fs, ref_lead='II', sig_name=sig_name)
    rr_ms = np.diff(rpeaks) / fs * 1000.0
    rr_med = float(np.median(rr_ms)) if len(rr_ms) else 0.0
    hr = 60000.0 / rr_med if rr_med else None

    out = {'fs': fs, 'n_rpeaks': int(len(rpeaks)), 'RR_ms': round(rr_med, 1),
           'HR_bpm': round(hr, 1) if hr else None}

    # II 导联做时序间期（清晰）
    ii = sig[:, _col(sig_name, 'II')]
    bts = delineate_lead(ii, rpeaks, fs)
    qrs_ms = [ (b['qrs_off']-b['qrs_on'])/fs*1000 for b in bts if (b['qrs_off']-b['qrs_on'])>0 ]
    pr_ms  = [ (b['qrs_on']-b['p_on'])/fs*1000 for b in bts if (b['qrs_on']-b['p_on'])>0 ]
    qt_ms  = [ (b['t_off']-b['qrs_on'])/fs*1000 for b in bts if (b['t_off']-b['qrs_on'])>0 ]
    out['QRS_ms'] = round(float(np.median(qrs_ms)),1) if qrs_ms else None
    out['PR_ms']  = round(float(np.median(pr_ms)),1)  if pr_ms  else None
    out['QT_ms']  = round(float(np.median(qt_ms)),1)  if qt_ms  else None
    out['QTc_ms'] = round(out['QT_ms']/math.sqrt(rr_med/1000.0),1) if (out['QT_ms'] and rr_med) else None

    # 逐导联 ST_Amp80ms（J+80ms 减基线）
    st = {}
    for lead in ORDER:
        try:
            li = _col(sig_name, lead)
        except ValueError:
            st[lead] = None; continue
        y = sig[:, li]
        bts_l = delineate_lead(y, rpeaks, fs)
        vals = []
        for b in bts_l:
            r = b['r']
            st_idx = min(r + int(0.150*fs), len(y)-1)   # 固定 R+150ms（ST 段平台），与 shift_st 抬高区对齐
            vals.append(float(y[st_idx]) - b['baseline'])
        st[lead] = round(float(np.median(vals)), 4) if vals else None
    out['ST_Amp80ms_mV'] = st

    # 电轴：用 I、III 主波（R 峰附近 max-min）净振幅近似
    def lead_amp(sig_name, lead):
        y = sig[:, _col(sig_name, lead)]
        amps=[]
        for r in rpeaks:
            a=r-int(0.080*fs); b=r+int(0.070*fs)
            a=max(0,a); seg=y[a:b+1]
            if len(seg): amps.append(float(np.max(seg)-np.min(seg)))
        return float(np.median(amps)) if amps else 0.0
    out['QRS_axis_deg'] = axis_from_i_iii(lead_amp(sig_name,'I'), lead_amp(sig_name,'III'))

    if verbose:
        out['_beats_II'] = bts
    return out