"""RawBoost augmentation (vendored reference implementation wrapper)."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class RawBoostCfg:
    """RawBoost hyperparameters."""

    N_f: int = 3
    nBands: int = 3
    minF: int = 20
    maxF: int = 8000
    minBW: int = 100
    maxBW: int = 500
    minCoeff: int = 10
    maxCoeff: int = 50
    minG: int = 0
    maxG: int = 0
    minBiasLinNonLin: int = 0
    maxBiasLinNonLin: int = 15
    P: int = 5
    g_sd: int = 1
    SNRmin: int = 0
    SNRmax: int = 20


def randRange(x1: float, x2: float, integer: int) -> float | int:
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    y_scalar = float(y.item())
    if integer:
        y_out: float | int = int(y_scalar)
    else:
        y_out = y_scalar
    return y_out


def normWav(x: np.ndarray, always: int) -> np.ndarray:
    if always:
        x = x / np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
        x = x / np.amax(abs(x))
    return x


def genNotchCoeffs(
    nBands: int,
    minF: int,
    maxF: int,
    minBW: int,
    maxBW: int,
    minCoeff: int,
    maxCoeff: int,
    minG: int,
    maxG: int,
    fs: int,
) -> np.ndarray:
    b: np.ndarray | float = 1
    for _ in range(0, nBands):
        fc = randRange(minF, maxF, 0)
        bw = randRange(minBW, maxBW, 0)
        c = randRange(minCoeff, maxCoeff, 1)

        if c / 2 == int(c / 2):
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000
        b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window="hamming", fs=fs), b)

    G = randRange(minG, maxG, 0)
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, G / 20) * b / np.amax(abs(h))
    return b


def filterFIR(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), "constant")
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N / 2) : int(y.shape[0] - N / 2)]
    return y


def LnL_convolutive_noise(
    x: np.ndarray,
    N_f: int,
    nBands: int,
    minF: int,
    maxF: int,
    minBW: int,
    maxBW: int,
    minCoeff: int,
    maxCoeff: int,
    minG: int,
    maxG: int,
    minBiasLinNonLin: int,
    maxBiasLinNonLin: int,
    fs: int,
) -> np.ndarray:
    y: np.ndarray = np.zeros_like(x, dtype=np.float64)
    min_g = minG
    max_g = maxG
    for i in range(0, N_f):
        if i == 1:
            min_g = min_g - minBiasLinNonLin
            max_g = max_g - maxBiasLinNonLin
        b = genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, min_g, max_g, fs)
        y = y + filterFIR(np.power(x, (i + 1)), b)
    y = y - np.mean(y)
    y = normWav(y, 0)
    return y.astype(np.float32)


def ISD_additive_noise(x: np.ndarray, P: int, g_sd: int) -> np.ndarray:
    beta = randRange(0, P, 0)

    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len * (beta / 100))
    p = np.random.permutation(x_len)[:n]
    f_r = np.multiply(((2 * np.random.rand(p.shape[0])) - 1), ((2 * np.random.rand(p.shape[0])) - 1))
    r = g_sd * x[p] * f_r
    y[p] = x[p] + r
    y = normWav(y, 0)
    return y.astype(np.float32)


def SSI_additive_noise(
    x: np.ndarray,
    SNRmin: int,
    SNRmax: int,
    nBands: int,
    minF: int,
    maxF: int,
    minBW: int,
    maxBW: int,
    minCoeff: int,
    maxCoeff: int,
    minG: int,
    maxG: int,
    fs: int,
) -> np.ndarray:
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise, 1)
    snr = randRange(SNRmin, SNRmax, 0)
    noise = noise / np.linalg.norm(noise, 2) * np.linalg.norm(x, 2) / 10.0 ** (0.05 * snr)
    x = x + noise
    return x.astype(np.float32)


def process_rawboost_feature(feature: np.ndarray, sr: int, cfg: RawBoostCfg, algo: int) -> np.ndarray:
    """Apply selected RawBoost algorithm(s)."""
    if algo == 1:
        feature = LnL_convolutive_noise(
            feature,
            cfg.N_f,
            cfg.nBands,
            cfg.minF,
            cfg.maxF,
            cfg.minBW,
            cfg.maxBW,
            cfg.minCoeff,
            cfg.maxCoeff,
            cfg.minG,
            cfg.maxG,
            cfg.minBiasLinNonLin,
            cfg.maxBiasLinNonLin,
            sr,
        )
    elif algo == 2:
        feature = ISD_additive_noise(feature, cfg.P, cfg.g_sd)
    elif algo == 3:
        feature = SSI_additive_noise(
            feature,
            cfg.SNRmin,
            cfg.SNRmax,
            cfg.nBands,
            cfg.minF,
            cfg.maxF,
            cfg.minBW,
            cfg.maxBW,
            cfg.minCoeff,
            cfg.maxCoeff,
            cfg.minG,
            cfg.maxG,
            sr,
        )
    elif algo == 4:
        feature = LnL_convolutive_noise(
            feature,
            cfg.N_f,
            cfg.nBands,
            cfg.minF,
            cfg.maxF,
            cfg.minBW,
            cfg.maxBW,
            cfg.minCoeff,
            cfg.maxCoeff,
            cfg.minG,
            cfg.maxG,
            cfg.minBiasLinNonLin,
            cfg.maxBiasLinNonLin,
            sr,
        )
        feature = ISD_additive_noise(feature, cfg.P, cfg.g_sd)
        feature = SSI_additive_noise(
            feature,
            cfg.SNRmin,
            cfg.SNRmax,
            cfg.nBands,
            cfg.minF,
            cfg.maxF,
            cfg.minBW,
            cfg.maxBW,
            cfg.minCoeff,
            cfg.maxCoeff,
            cfg.minG,
            cfg.maxG,
            sr,
        )
    elif algo == 5:
        feature = LnL_convolutive_noise(
            feature,
            cfg.N_f,
            cfg.nBands,
            cfg.minF,
            cfg.maxF,
            cfg.minBW,
            cfg.maxBW,
            cfg.minCoeff,
            cfg.maxCoeff,
            cfg.minG,
            cfg.maxG,
            cfg.minBiasLinNonLin,
            cfg.maxBiasLinNonLin,
            sr,
        )
        feature = ISD_additive_noise(feature, cfg.P, cfg.g_sd)
    elif algo == 6:
        feature = LnL_convolutive_noise(
            feature,
            cfg.N_f,
            cfg.nBands,
            cfg.minF,
            cfg.maxF,
            cfg.minBW,
            cfg.maxBW,
            cfg.minCoeff,
            cfg.maxCoeff,
            cfg.minG,
            cfg.maxG,
            cfg.minBiasLinNonLin,
            cfg.maxBiasLinNonLin,
            sr,
        )
        feature = SSI_additive_noise(
            feature,
            cfg.SNRmin,
            cfg.SNRmax,
            cfg.nBands,
            cfg.minF,
            cfg.maxF,
            cfg.minBW,
            cfg.maxBW,
            cfg.minCoeff,
            cfg.maxCoeff,
            cfg.minG,
            cfg.maxG,
            sr,
        )
    elif algo == 7:
        feature = ISD_additive_noise(feature, cfg.P, cfg.g_sd)
        feature = SSI_additive_noise(
            feature,
            cfg.SNRmin,
            cfg.SNRmax,
            cfg.nBands,
            cfg.minF,
            cfg.maxF,
            cfg.minBW,
            cfg.maxBW,
            cfg.minCoeff,
            cfg.maxCoeff,
            cfg.minG,
            cfg.maxG,
            sr,
        )
    elif algo == 8:
        feature1 = LnL_convolutive_noise(
            feature,
            cfg.N_f,
            cfg.nBands,
            cfg.minF,
            cfg.maxF,
            cfg.minBW,
            cfg.maxBW,
            cfg.minCoeff,
            cfg.maxCoeff,
            cfg.minG,
            cfg.maxG,
            cfg.minBiasLinNonLin,
            cfg.maxBiasLinNonLin,
            sr,
        )
        feature2 = ISD_additive_noise(feature, cfg.P, cfg.g_sd)
        feature = normWav(feature1 + feature2, 0)
    return feature.astype(np.float32)
