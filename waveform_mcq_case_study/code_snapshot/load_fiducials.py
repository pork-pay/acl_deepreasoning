#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逐拍 fiducial 定位（轻量，不依赖 FeatureDB 重环境）。

策略：
  - R 峰：neurokit2 ecg_peaks（在参考导联 II 上检测，12 导联同步采集 → 共享给所有导联）。
  - P/QRS/T 边界：在 R 峰邻域用纯 numpy 分割（基线、门限、斜率），失败回退到生理时序先验。
    保证扰动前/后用同一套口径测量，apples-to-apples 比较。

公开 API:
  detect_rpeaks(sig, fs, ref_lead='II') -> np.ndarray  (R 峰样本索引)
  delineate_lead(sig_lead, rpeaks, fs) -> list[Beat]   每拍 fiducial 样本索引
  Beat 为 dict: p_on,p_off,qrs_on,qrs_off,j_point,t_on,t_peak,t_off,baseline
"""
import numpy as np
import neurokit2 as nk

# 生理时序先验（相对 R 峰，毫秒），回退用
PRIOR = {
    "p_on_off_r":  -200, "p_off_off_r": -110,          # P 波
    "qrs_on_off_r": -80, "qrs_off_off_r":  70,         # QRS（J 点 = qrs_off）
    "t_on_off_r":  150, "t_off_off_r": 380,            # T 波
}


def _baseline(sig_lead, r, fs, span_ms=30):
    """取 R 前一段(TP 段)中位做等电位基线。"""
    a = max(0, r - int(250/1000*fs))
    b = max(1, r - int(220/1000*fs))
    return float(np.median(sig_lead[a:b]))


def detect_rpeaks(sig, fs, ref_lead='II', sig_name=None):
    """参考导联检测 R 峰。sig:(N,12) mV, sig_name 大写列表。返回 R 峰样本索引。"""
    names = [s.upper() for s in (sig_name or [])]
    if ref_lead.upper() in names:
        lead = sig[:, names.index(ref_lead.upper())]
    else:
        lead = sig[:, 0]
    clean = nk.ecg_clean(lead, sampling_rate=fs)
    _, info = nk.ecg_peaks(clean, sampling_rate=fs)
    r = info['ECG_R_Peaks']
    # 过滤过近 (<0.40 RR) 的误检(双计伪峰); 留 >0.40RR 的早搏(PVC/PAC)不被误删
    if len(r) >= 3:
        med = np.median(np.diff(r))
        keep = [r[0]]
        for p in r[1:]:
            if p - keep[-1] >= med * 0.40:
                keep.append(p)
        r = np.array(keep, dtype=int)
    # 幅度过滤: 拒掉局部形变远小于 QRS 中位幅度的伪峰(平直潜伏段/滤窗振铃产生的误检)
    if len(r) >= 3:
        w = int(0.030*fs)
        amps = []
        for p in r:
            a, b = max(0, p-w), min(len(clean)-1, p+w)
            amps.append(float(np.max(clean[a:b+1]) - np.min(clean[a:b+1])))
        amps = np.array(amps)
        med_amp = np.median(amps)
        if med_amp > 0:
            r = r[amps > np.maximum(0.45*med_amp, 0.35)]
    return r
    return r


def _cross_at(sig, idx):
    idx = int(min(max(idx, 0), len(sig) - 1))
    return float(sig[idx])


def delineate_lead(sig_lead, rpeaks, fs):
    """
    单导联逐拍 fiducial。返回 list[dict]，每个 dict 含:
      p_on,p_off,qrs_on,qrs_off,j_point(=qrs_off),t_on,t_peak,t_off,baseline
    所有值为 fs 坐标系样本索引。
    """
    beats = []
    x = np.asarray(sig_lead, dtype=float)
    n = len(x)
    for i, r in enumerate(rpeaks):
        r = int(r)
        baseline = _baseline(x, r, fs)

        # ---- QRS：在 R 邻域用门限定 on/off ----
        win_q_on = r + int(PRIOR['qrs_on_off_r']/1000*fs)   # -80ms
        win_q_off = r + int(0.060*fs)                        # +60ms（留出 ST 编辑空间，防 QRS 测量被 ST 抬高污染）
        seg = x[win_q_on:win_q_off+1] - baseline
        thr = max(0.05 * np.max(np.abs(seg)), 0.05)
        # qrs_on: 从 win_q_on 向右找首个超门限
        qrs_on = win_q_on
        for k in range(len(seg)):
            if abs(seg[k]) > thr:
                qrs_on = win_q_on + k; break
        # qrs_off: 从 win_q_off 向左找最后一个超门限
        qrs_off = win_q_off
        for k in range(len(seg)-1, -1, -1):
            if abs(seg[k]) > thr:
                qrs_off = win_q_on + k; break

        # ---- P 波：锚到 qrs_on (而非 R)——QRS 右移(如 1°AVB 延长 PR)时仍能找到没动的 P ----
        p_b = qrs_on - int(0.040*fs)
        p_a = max(0, p_b - int(0.260*fs))
        # 防抓前一拍 T 波当 P：把 P 搜索下界抬到前一拍 T_off 之后
        if i > 0 and beats[-1].get('t_off', -1) >= p_a:
            p_a = max(p_a, int(beats[-1]['t_off']) + max(1, int(0.040*fs)))
            if p_a >= p_b - 2:
                p_a = max(0, p_b - int(0.150*fs))
        pseg = x[p_a:p_b+1] - baseline
        p_on, p_off = p_a, p_b
        if len(pseg) > 5 and np.max(np.abs(pseg)) > 0.03:
            pthr = 0.4 * np.max(np.abs(pseg))
            for k in range(len(pseg)):
                if abs(pseg[k]) > pthr: p_on = p_a+k; break
            for k in range(len(pseg)-1,-1,-1):
                if abs(pseg[k]) > pthr: p_off = p_a+k; break

        # ---- T 波：J+120 .. R+430ms 窗内找峰与回基线 ----
        t_a = qrs_off + int(0.080*fs)        # ST 后开始找 T
        t_b = min(n-1, r + int(0.430*fs))
        tseg = x[t_a:t_b+1] - baseline
        t_on, t_peak, t_off = t_a, t_a, t_b
        if len(tseg) > 5 and np.max(np.abs(tseg)) > 0.03:
            t_peak = t_a + int(np.argmax(np.abs(tseg)))
            amp = x[t_peak] - baseline
            tthr = 0.25 * abs(amp)
            if amp >= 0:
                cross = np.where(tseg >= tthr)[0]
            else:
                cross = np.where(tseg <= -tthr)[0]
            if len(cross):
                t_on = t_a + cross[0]
                t_off_idx = cross[-1]
                # t_off 之后找回基线
                tail = tseg[t_off_idx:]
                tt = np.where(np.abs(tail) < tthr*0.5)[0]
                t_off = t_a + t_off_idx + (tt[0] if len(tt) else len(tail)-1)
            else:
                t_on = t_a; t_off = t_b
        else:
            t_on = r + int(PRIOR['t_on_off_r']/1000*fs)
            t_off = r + int(PRIOR['t_off_off_r']/1000*fs)

        beats.append({
            'r': r, 'p_on': p_on, 'p_off': p_off,
            'qrs_on': qrs_on, 'qrs_off': qrs_off, 'j_point': qrs_off,
            't_on': t_on, 't_peak': t_peak, 't_off': min(t_off, n-1),
            'baseline': baseline,
        })
    return beats