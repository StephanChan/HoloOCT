# -*- coding: utf-8 -*-
"""
XZ 2D Off-Axis Full-Field OCT Simulation
=========================================
Simulates the off-axis full-field FD-OCT principle in 2D (X vs Z).
Includes the full processing pipeline from the paper (Hillmann et al. 2013).

Pipeline:
  I(x, lambda)          -- raw interference image (X-lambda plane)
    → Add camera noise  -- shot noise (Poisson) + readout noise (Gaussian)
    → FFT along X       -- spatial frequency domain (kx-lambda)
    → bandpass filter    -- select one sideband, remove DC & conjugate
    → IFFT along X       -- filtered complex signal in X-lambda
    → FFT along lambda   -- full-depth XZ reconstruction (no mirror)

Uses CuPy for GPU acceleration.

Parameters:
  λ₀ = 840 nm, Δλ = 40 nm
  NA = 0.05, M = 0.62
  Sample FOV: 16 mm (X), Camera: 1100 × 1600 pixels at 9 μm
  Off-axis reference tilt: 2°
  Full well depth: 100,000 e- (60% fill)
  Readout noise: 10 e- RMS
"""

import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
import time
import os
import json
import hashlib

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================

CENTER_WAVELENGTH = 840e-9
WAVELENGTH_RANGE   = 40e-9
LAMBDA_START = CENTER_WAVELENGTH - WAVELENGTH_RANGE / 2
LAMBDA_END   = CENTER_WAVELENGTH + WAVELENGTH_RANGE / 2

SAMPLE_FOV_X = 16.0e-3

NA = 0.01
w0_sample = CENTER_WAVELENGTH / (np.pi * NA)
zR_sample = np.pi * w0_sample**2 / CENTER_WAVELENGTH

REFERENCE_Z = 2.0e-3
FOCUS_Z = 2.0e-3

N_SPECTRAL = 1600
N_X        = 1100
PIXEL_SIZE = 9e-6
CAMERA_FOV_X = N_X * PIXEL_SIZE

# Camera noise
FULL_WELL_DEPTH = 100000
WELL_FILL_FRACTION = 0.60
MAX_SIGNAL_E = FULL_WELL_DEPTH * WELL_FILL_FRACTION
READOUT_NOISE_E = 10.0

# Optical system
M  = CAMERA_FOV_X / SAMPLE_FOV_X
SAMPLE_PIXEL_SIZE = PIXEL_SIZE / M
theta_max_deg = np.degrees(np.arcsin(CENTER_WAVELENGTH / (2 * PIXEL_SIZE)))

# Reference tilt
REF_TILT_ANGLE_DEG = 1.3
REF_TILT_ANGLE = REF_TILT_ANGLE_DEG * np.pi / 180
SIDEBAND_SIGN = +1
SIDEBAND_HALF_WIDTH_FRACTION = 0.65
SIDEBAND_TAPER_FRACTION = 0.3
KEEP_HIGH_FREQUENCY_TAIL = False
SUBTRACT_COMMON_MODE_SPECTRUM = False
SPECTRAL_WINDOW = "hann"  # "hann", "hamming", "blackman", or None
REF_AMP = 1.0
SAMPLE_FIELD_SCALE = 0.03
INCLUDE_SAMPLE_SELF_INTERFERENCE = False
PHANTOM_VERSION = 2

# Phantom
N_SCATTERERS = 600
SCATTER_X_RANGE = SAMPLE_FOV_X * 0.8
SCATTER_Z_RANGE = 3.0e-3
SCATTER_MIN_REFL = 0.5
SCATTER_MAX_REFL = 1.0

# Derived
k_start = 2 * np.pi / LAMBDA_END
k_end   = 2 * np.pi / LAMBDA_START
dk = (k_end - k_start) / (N_SPECTRAL - 1)
k_array = np.linspace(k_start, k_end, N_SPECTRAL)

DELTA_Z = 0.44 * CENTER_WAVELENGTH**2 / WAVELENGTH_RANGE
Z_MAX = np.pi / (2 * dk)
DISPLAY_DC_EXCLUDE_MM = 0.20

fringe_period_um = CENTER_WAVELENGTH / np.sin(REF_TILT_ANGLE) * 1e6
fringe_period_px = fringe_period_um / (PIXEL_SIZE * 1e6)
kx_nyquist = np.pi / PIXEL_SIZE
kx_carrier = 2 * np.pi * np.sin(REF_TILT_ANGLE) / CENTER_WAVELENGTH
kx_sample_cutoff = 2 * np.pi * NA / (M * CENTER_WAVELENGTH)

print("=" * 67)
print("XZ Off-Axis FF-OCT Simulation -- Optical System Parameters")
print("=" * 67)
print(f"  λ₀ = {CENTER_WAVELENGTH*1e9:.1f} nm")
print(f"  Δλ = {WAVELENGTH_RANGE*1e9:.1f} nm")
print(f"  NA = {NA:.3f}")
print(f"  w₀ = {w0_sample*1e6:.2f} μm")
print(f"  zR = {zR_sample*1e6:.1f} μm")
print(f"  M  = {M:.3f}")
print(f"  Sample FOV X: {SAMPLE_FOV_X*1e3:.1f} mm")
print(f"  Camera FOV X: {CAMERA_FOV_X*1e3:.2f} mm")
print(f"  Sample pixel: {SAMPLE_PIXEL_SIZE*1e6:.2f} μm")
print(f"  δz = {DELTA_Z*1e6:.2f} μm")
print(f"  z_max = {Z_MAX*1e3:.2f} mm")
print(f"  θ_tilt = {REF_TILT_ANGLE_DEG:.2f}°")
print(f"  θ_max (Nyquist) = {theta_max_deg:.2f}°")
print(f"  Fringe period = {fringe_period_um:.1f} μm ({fringe_period_px:.1f} px)")
print(f"  kx carrier / Nyquist = {kx_carrier/kx_nyquist:.2f}")
print(f"  est. sample kx cutoff / Nyquist = {kx_sample_cutoff/kx_nyquist:.2f}")
if kx_sample_cutoff >= kx_carrier or kx_carrier + kx_sample_cutoff >= kx_nyquist:
    print("  WARNING: sidebands are not cleanly separated with these NA/M/pixel settings.")
print(f"  Reference amplitude: {REF_AMP:.3g}")
print(f"  Sample field scale:  {SAMPLE_FIELD_SCALE:.3g}")
print(f"  Include |E_sample|^2: {INCLUDE_SAMPLE_SELF_INTERFERENCE}")
print(f"  Sideband taper fraction: {SIDEBAND_TAPER_FRACTION:.2f}")
print(f"  Keep high-frequency tail: {KEEP_HIGH_FREQUENCY_TAIL}")
print(f"  Subtract common-mode spectrum: {SUBTRACT_COMMON_MODE_SPECTRUM}")
print(f"  Spectral window: {SPECTRAL_WINDOW}")
print(f"  Phantom version: {PHANTOM_VERSION}")
print(f"  --- Reference & Focus ---")
print(f"  Reference plane (OCT z=0): {REFERENCE_Z*1e3:.1f} mm physical depth")
print(f"  Objective focus:           {FOCUS_Z*1e3:.1f} mm physical depth")
print(f"  OCT depth range:           ±{SCATTER_Z_RANGE*1e3:.1f} mm")
print(f"  Physical depth range:      [{((REFERENCE_Z-SCATTER_Z_RANGE)*1e3):.1f}, "
      f"{((REFERENCE_Z+SCATTER_Z_RANGE)*1e3):.1f}] mm")
print(f"  --- Camera Noise ---")
print(f"  Full well depth:       {FULL_WELL_DEPTH} e-")
print(f"  Well fill fraction:    {WELL_FILL_FRACTION*100:.0f}%")
print(f"  Max signal:            {MAX_SIGNAL_E:.0f} e-")
print(f"  Readout noise (RMS):   {READOUT_NOISE_E:.1f} e-")
print("=" * 67)


def generate_eye_phantom():
    """Generate anterior segment eye phantom in OCT-depth coordinates."""
    rng = np.random.default_rng(42)
    
    CORNEA_APEX_OCT = -1.5e-3
    CORNEA_THICKNESS = 0.55e-3
    CORNEA_ANT_RADIUS = 7.8e-3
    CORNEA_POST_RADIUS = 6.5e-3
    CORNEA_WIDTH = 6.0e-3
    AC_DEPTH = 3.0e-3
    IRIS_OCT = CORNEA_APEX_OCT + CORNEA_THICKNESS + AC_DEPTH
    PUPIL_RADIUS = 2.0e-3
    IRIS_OUTER_RADIUS = 3.2e-3
    LENS_APEX_OCT = IRIS_OCT + 0.18e-3
    LENS_ANT_RADIUS = 10.0e-3
    LENS_THICKNESS = 0.75e-3
    LENS_APERTURE = 4.2e-3

    N_CORNEA_ANT = 2500
    N_CORNEA_POST = 2500
    N_CORNEA_TISSUE = 2000
    N_IRIS = 3000
    N_LENS_ANT = 2000
    N_LENS_TISSUE = 1500

    x_s_list, z_s_list, a_s_list = [], [], []

    REFL_CORNEA_SURF = 2.0
    REFL_CORNEA_STR = 0.3
    REFL_IRIS = 1.0
    REFL_LENS_CAP = 0.8
    REFL_LENS_STR = 0.4

    def spherical_surface_z(x, z_apex, radius):
        return z_apex + radius - np.sqrt(np.maximum(radius**2 - x**2, 0))

    def append_points(x, z, a):
        x_s_list.append(x)
        z_s_list.append(z)
        a_s_list.append(a)

    # Cornea: anterior and posterior spherical surfaces plus stromal scatterers
    # constrained between the two surfaces.
    x = rng.uniform(-CORNEA_WIDTH/2, CORNEA_WIDTH/2, N_CORNEA_ANT)
    z = spherical_surface_z(x, CORNEA_APEX_OCT, CORNEA_ANT_RADIUS)
    z += rng.normal(0, 2e-6, N_CORNEA_ANT)
    a = REFL_CORNEA_SURF * rng.uniform(0.7, 1.0, N_CORNEA_ANT)
    append_points(x, z, a)

    x = rng.uniform(-CORNEA_WIDTH/2, CORNEA_WIDTH/2, N_CORNEA_POST)
    z = spherical_surface_z(x, CORNEA_APEX_OCT + CORNEA_THICKNESS,
                            CORNEA_POST_RADIUS)
    z += rng.normal(0, 2e-6, N_CORNEA_POST)
    a = REFL_CORNEA_SURF * 0.8 * rng.uniform(0.7, 1.0, N_CORNEA_POST)
    append_points(x, z, a)

    x = rng.uniform(-CORNEA_WIDTH*0.45, CORNEA_WIDTH*0.45, N_CORNEA_TISSUE)
    z_ant = spherical_surface_z(x, CORNEA_APEX_OCT, CORNEA_ANT_RADIUS)
    z_post = spherical_surface_z(x, CORNEA_APEX_OCT + CORNEA_THICKNESS,
                                 CORNEA_POST_RADIUS)
    u = rng.uniform(0.08, 0.92, N_CORNEA_TISSUE)
    z = z_ant + u * (z_post - z_ant) + rng.normal(0, 3e-6, N_CORNEA_TISSUE)
    a = REFL_CORNEA_STR * rng.uniform(0.25, 1.0, N_CORNEA_TISSUE)
    append_points(x, z, a)

    # Iris: two textured leaflets outside the pupil.  The center is left open.
    n_half = N_IRIS // 2
    xil = rng.uniform(-IRIS_OUTER_RADIUS, -PUPIL_RADIUS, n_half)
    xir = rng.uniform(PUPIL_RADIUS, IRIS_OUTER_RADIUS, N_IRIS - n_half)
    xi = np.concatenate([xil, xir])
    normalized = (np.abs(xi) - PUPIL_RADIUS) / (IRIS_OUTER_RADIUS - PUPIL_RADIUS)
    zi = IRIS_OCT + 0.18e-3 * (1.0 - normalized)
    zi += rng.uniform(0, 0.12e-3, len(xi)) + rng.normal(0, 12e-6, len(xi))
    ai = REFL_IRIS * rng.uniform(0.5, 1.0, len(xi))
    append_points(xi, zi, ai)

    # Lens: anterior capsule and tissue volume behind the capsule.
    x = rng.uniform(-LENS_APERTURE/2, LENS_APERTURE/2, N_LENS_ANT)
    z = spherical_surface_z(x, LENS_APEX_OCT, LENS_ANT_RADIUS)
    z += rng.normal(0, 2e-6, N_LENS_ANT)
    a = REFL_LENS_CAP * rng.uniform(0.7, 1.0, N_LENS_ANT)
    append_points(x, z, a)

    x = rng.uniform(-LENS_APERTURE*0.45, LENS_APERTURE*0.45, N_LENS_TISSUE)
    z_capsule = spherical_surface_z(x, LENS_APEX_OCT, LENS_ANT_RADIUS)
    lens_depth = LENS_THICKNESS * np.sqrt(
        np.maximum(1.0 - (2*x/LENS_APERTURE)**2, 0.0)
    )
    z = z_capsule + rng.uniform(0.08, 0.95, N_LENS_TISSUE) * lens_depth
    z += rng.normal(0, 5e-6, N_LENS_TISSUE)
    a = REFL_LENS_STR * rng.uniform(0.25, 1.0, N_LENS_TISSUE)
    append_points(x, z, a)

    x_s = np.concatenate(x_s_list)
    z_s = np.concatenate(z_s_list)
    a_s = np.concatenate(a_s_list)
    idx = rng.permutation(len(x_s))
    x_s, z_s, a_s = x_s[idx], z_s[idx], a_s[idx]

    print(f"  Eye phantom: {len(x_s)} scatterers")
    return x_s, z_s, a_s


def compute_sample_field_gpu(x_sample_s, z_s, a_s, x_cam_gpu, k_gpu):
    """Compute sample E-field at camera plane (GPU)."""
    N_scat = len(x_sample_s)
    w0_g  = cp.float32(w0_sample)
    zR_g  = cp.float32(zR_sample)
    Mg    = cp.float32(M)
    x_cam_s = cp.asarray(x_sample_s, dtype=cp.float32) * Mg
    Etot = cp.zeros((N_SPECTRAL, N_X), dtype=cp.complex64)

    for idx in range(N_scat):
        xsc = x_cam_s[idx]
        zs  = z_s[idx]
        amp = a_s[idx]
        dx  = x_cam_gpu - xsc
        zr  = zs / zR_g
        wz  = Mg * w0_g * cp.sqrt(1.0 + zr**2)
        ge  = cp.exp(-dx**2 / wz**2)
        ap  = cp.exp(-1j * 2.0 * k_gpu * zs)
        if abs(zs) > 0.1 * zR_g:
            Rz = zs * (1.0 + (zR_g / zs)**2)
            df = cp.exp(-1j * k_gpu[:, None] * (dx[None,:]/Mg)**2 / (2.0 * Rz))
        else:
            df = cp.ones((N_SPECTRAL, N_X), dtype=cp.complex64)
        Etot += amp * ge[None,:] * ap[:,None] * df
    return Etot


def add_camera_noise(I_cpu, max_signal_e=MAX_SIGNAL_E,
                     readout_noise_e=READOUT_NOISE_E, rng_seed=123):
    """Add shot noise (Poisson) + readout noise (Gaussian)."""
    rng = np.random.default_rng(rng_seed)
    I_elec = I_cpu * max_signal_e
    shot = rng.poisson(np.maximum(I_elec, 0))
    readout = rng.normal(0, readout_noise_e, I_elec.shape)
    I_noisy = np.maximum(shot.astype(np.float64) + readout.astype(np.float64), 0)
    noise_info = {
        "max_signal_e": max_signal_e,
        "mean_signal_e": float(np.mean(I_elec)),
        "readout_noise_rms_e": readout_noise_e,
        "total_noise_rms_e": float(np.sqrt(np.mean((I_noisy - I_elec)**2))),
        "SNR_dB": float(20 * np.log10(np.mean(I_elec) / 
                      np.sqrt(np.mean((I_noisy - I_elec)**2)) + 1e-10))
    }
    return I_noisy.astype(np.float32), noise_info


def plot_phantom(xs, zs, a):
    """Figure 1: Ground truth phantom."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(zs*1e3, xs*1e3, c=a, s=np.clip(a*5, 1, 10),
                    cmap='jet', alpha=0.6, edgecolors='none')
    ax.set_xlabel('OCT Depth z_oct [mm]'); ax.set_ylabel('Sample X [mm]')
    ax.set_title(f'Eye Phantom ({len(xs)} scatterers)\nRef plane={REFERENCE_Z*1e3:.1f} mm')
    ax.set_xlim(-SCATTER_Z_RANGE*1e3, SCATTER_Z_RANGE*1e3)
    ax.set_ylim(-SCATTER_X_RANGE/2*1e3, SCATTER_X_RANGE/2*1e3)
    ax.grid(True, alpha=0.2)
    ax2 = ax.twiny()
    ticks = np.linspace(-SCATTER_Z_RANGE*1e3, SCATTER_Z_RANGE*1e3, 5)
    ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{REFERENCE_Z*1e3 + t:.1f}" for t in ticks])
    ax2.set_xlabel('Physical depth [mm]')
    plt.colorbar(sc, ax=ax, label='Reflectivity')
    ax.axvline(0, color='green', ls='--', lw=1.5, alpha=0.7, label='Ref plane')
    for xy, txt, xyt in [
        ((-1.5, 0), 'Cornea\n(epi)', (-2.2, 4)),
        ((-0.95, 0.5), 'Cornea\n(endo)', (-0.7, 4.5)),
        ((-0.2, 0), 'Ant.\nchamber', (0.5, 4)),
        ((1.95, -3), 'Iris', (2.5, -5)),
        ((2.05, 1.5), 'Lens\n(capsule)', (2.6, 2.5)),
    ]:
        ax.annotate(txt, xy=xy, xytext=xyt,
                    arrowprops=dict(arrowstyle='->', color='white', alpha=0.7),
                    color='white', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='k', alpha=0.6))
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout(); plt.show()
    return fig


def plot_ground_truth_xz(xs, zs, a):
    """Ground-truth scatterer map using the same XZ axes as reconstruction."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(zs*1e3, xs*1e3, color='cyan', s=np.clip(a*4, 1, 8),
               alpha=0.65, edgecolors='none')
    ax.set_xlabel('Z [mm]')
    ax.set_ylabel('X [mm]')
    ax.set_title('Ground Truth Phantom')
    ax.set_xlim(-Z_MAX*1e3, Z_MAX*1e3)
    ax.set_ylim(-SAMPLE_FOV_X/2*1e3, SAMPLE_FOV_X/2*1e3)
    ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.show()
    return fig


def generate_interference(xs, zs, a, tilt_angle, spectral_window=True):
    """
    Generate raw interference I(x, lambda) on GPU.
    
    KEY FIX: Reference amplitude is boosted to make fringes visible.
    With REF_AMP = 10:
      |R|² = 100 (DC)
      cross term 2*Re(R*E) ≈ ±20 * |E|  (up to ±100 at bright pixels)
      This creates strong, clearly visible fringes in the interference.
    """
    ref_amp = cp.float32(REF_AMP)
    x_cam = np.linspace(-CAMERA_FOV_X/2, CAMERA_FOV_X/2, N_X)
    xcg = cp.asarray(x_cam, dtype=cp.float32)
    kg  = cp.asarray(k_array, dtype=cp.float32)
    zsg = cp.asarray(zs, dtype=cp.float32)
    ag  = cp.asarray(a, dtype=cp.float32)
    
    E = cp.float32(SAMPLE_FIELD_SCALE) * compute_sample_field_gpu(xs, zsg, ag, xcg, kg)
    
    # Reference: plane wave at tilt angle
    R = ref_amp * cp.exp(1j * (kg[:,None] * xcg[None,:] * cp.sin(cp.float32(tilt_angle))))
    
    # Interference: I = |R+E|² = |R|² + |E|² + 2*Re(R*·E)
    if INCLUDE_SAMPLE_SELF_INTERFERENCE:
        I = cp.abs(R + E)**2
    else:
        I = cp.abs(R)**2 + 2 * cp.real(R * cp.conj(E))
    
    # Print diagnostics
    em = float(cp.mean(cp.abs(E)))
    emx = float(cp.max(cp.abs(E)))
    dc = float(cp.mean(cp.abs(R)**2))
    im = float(cp.mean(I))
    print(f"    |R|²(mean)={dc:.1f}, |E|(mean)={em:.4f}, |E|(max)={emx:.3f}")
    print(f"    I(mean)={im:.3f}, cross term amplitude ~{2*REF_AMP:.2g}*|E|")
    print(f"    max |E|^2 / max cross term ~{emx/(2*REF_AMP + 1e-12):.3f}")
    
    if spectral_window:
        k0 = cp.float32(2 * np.pi / CENTER_WAVELENGTH)
        sk = cp.float32(2 * np.pi * WAVELENGTH_RANGE / CENTER_WAVELENGTH**2 / 2.355)
        I *= cp.exp(-(kg - k0)**2 / (2 * sk**2))[:, None]
    
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(I), x_cam


def plot_interference(I, x_cam, title_suffix="", noise_info=None):
    """Figure 2: Raw interference image."""
    l_ax = np.linspace(LAMBDA_START, LAMBDA_END, N_SPECTRAL) * 1e9
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    
    axs[0].imshow(I, aspect='auto', cmap='gray',
                  extent=[x_cam[0]*1e3, x_cam[-1]*1e3, l_ax[0], l_ax[-1]], origin='lower')
    axs[0].set_xlabel('Camera X [mm]'); axs[0].set_ylabel('λ [nm]')
    axs[0].set_title(f'I(x,λ) {title_suffix}')
    
    lz, xz = slice(0, 200), slice(N_X//2-30, N_X//2+30)
    axs[1].imshow(I[lz, xz], aspect='auto', cmap='gray',
                  extent=[x_cam[xz.start]*1e3, x_cam[xz.stop-1]*1e3,
                          l_ax[lz.start], l_ax[lz.stop-1]], origin='lower')
    axs[1].set_xlabel('Camera X [mm]'); axs[1].set_ylabel('λ [nm]')
    axs[1].set_title('Zoom: Fringes')
    
    cx, ox = N_X // 2, N_X // 4
    axs[2].plot(l_ax, I[:, cx], 'b-', lw=0.8, label=f'X={x_cam[cx]*1e3:.1f} mm')
    axs[2].plot(l_ax, I[:, ox], 'r--', lw=0.8, label=f'X={x_cam[ox]*1e3:.1f} mm')
    axs[2].set_xlabel('λ [nm]'); axs[2].set_ylabel('Intensity')
    axs[2].set_title('Spectra'); axs[2].legend(fontsize=8); axs[2].grid(True, alpha=0.3)
    
    if noise_info:
        txt = (f"Noise:\n  Max sig: {noise_info['max_signal_e']:.0f} e-\n"
               f"  Readout RMS: {noise_info['readout_noise_rms_e']:.1f} e-\n"
               f"  Total RMS: {noise_info['total_noise_rms_e']:.1f} e-\n"
               f"  SNR: {noise_info['SNR_dB']:.1f} dB")
        axs[0].text(0.02, 0.02, txt, transform=axs[0].transAxes,
                    fontsize=8, color='white', va='bottom',
                    bbox=dict(boxstyle='round', fc='k', alpha=0.7))
    
    plt.tight_layout(); plt.show()
    return fig


def plot_spatial_fft(I, tilt_angle, title_suffix=""):
    """Figure 3: Spatial FFT along X (kx-lambda)."""
    F = np.fft.fftshift(np.fft.fft(I, axis=1), axes=1)
    kx = np.fft.fftshift(np.fft.fftfreq(N_X, d=PIXEL_SIZE)) * 2 * np.pi
    kxu = kx / 1e6
    l_ax = np.linspace(LAMBDA_START, LAMBDA_END, N_SPECTRAL) * 1e9
    
    Fm = np.abs(F)
    Fd = np.zeros_like(Fm)
    for i in range(N_SPECTRAL):
        r = Fm[i, :]
        if r.max() > 0: Fd[i, :] = 20 * np.log10(r / r.max() + 1e-10)
    
    ck = 2 * np.pi * np.sin(tilt_angle) / CENTER_WAVELENGTH
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    im = axs[0,0].imshow(Fd, aspect='auto', vmin=-30, vmax=0,
                          extent=[kxu[0], kxu[-1], l_ax[0], l_ax[-1]],
                          origin='lower', cmap='hot')
    axs[0,0].set_xlabel('kx [rad/μm]'); axs[0,0].set_ylabel('λ [nm]')
    axs[0,0].set_title(f'|FFT_x(I)|² (dB/row) {title_suffix}')
    plt.colorbar(im, ax=axs[0,0], fraction=0.046)
    axs[0,0].axvline(0, color='cyan', ls=':', lw=0.5, alpha=0.7)
    if abs(tilt_angle) > 1e-6:
        axs[0,0].axvline(ck/1e6, color='lime', ls='--', lw=1.5, alpha=0.8,
                         label=f'Carrier ±{ck/1e6:.2f}')
        axs[0,0].axvline(-ck/1e6, color='lime', ls='--', lw=1.5, alpha=0.8)
        axs[0,0].legend(fontsize=7, loc='upper left')
    else:
        axs[0,0].text(0.5, 0.95, 'On-axis: carrier at kx=0\n(overlaps DC)',
                      transform=axs[0,0].transAxes, fontsize=8, color='white',
                      ha='center', va='top',
                      bbox=dict(boxstyle='round', fc='k', alpha=0.6))
    
    ci = N_SPECTRAL // 2
    axs[0,1].plot(kxu, Fd[ci, :], 'b-', lw=0.8)
    axs[0,1].set_xlabel('kx [rad/μm]'); axs[0,1].set_ylabel('|FFT| [dB]')
    axs[0,1].set_title(f'kx at λ={l_ax[ci]:.1f} nm')
    axs[0,1].grid(True, alpha=0.3)
    axs[0,1].axvline(ck/1e6, color='r', ls='--', alpha=0.7)
    axs[0,1].axvline(-ck/1e6, color='r', ls='--', alpha=0.7)
    
    kxp = Fm.sum(axis=0)
    axs[1,0].plot(kxu, kxp, 'b-', lw=0.8)
    axs[1,0].set_xlabel('kx [rad/μm]'); axs[1,0].set_ylabel('Integrated')
    axs[1,0].set_title('Integrated kx spectrum')
    axs[1,0].grid(True, alpha=0.3)
    axs[1,0].axvline(ck/1e6, color='r', ls='--', alpha=0.7)
    axs[1,0].axvline(-ck/1e6, color='r', ls='--', alpha=0.7)
    
    step = 8
    X, Y = np.meshgrid(kxu[::step], l_ax[::step])
    im = axs[1,1].pcolormesh(X, Y, Fd[::step, ::step], shading='auto',
                              cmap='hot', vmin=-30, vmax=0)
    axs[1,1].set_xlabel('kx [rad/μm]'); axs[1,1].set_ylabel('λ [nm]')
    axs[1,1].set_title('kx-λ map')
    plt.colorbar(im, ax=axs[1,1], fraction=0.046)
    
    plt.tight_layout(); plt.show()
    return fig, F, kx


def filter_spatial_sideband(F, kx_axis, tilt_angle, sideband_sign=SIDEBAND_SIGN):
    """Keep one off-axis cross-term sideband in kx."""
    ck = 2 * np.pi * np.sin(tilt_angle) / CENTER_WAVELENGTH
    bc = abs(ck)
    if bc <= 0:
        raise ValueError("Off-axis sideband filtering requires a non-zero reference tilt.")
    bh = bc * SIDEBAND_HALF_WIDTH_FRACTION
    taper = np.clip(SIDEBAND_TAPER_FRACTION, 0.0, 0.95) * bh
    
    center = np.sign(sideband_sign) * bc
    conjugate_mask = np.abs(kx_axis + center) < bh
    weights = np.zeros_like(kx_axis, dtype=np.float32)

    if KEEP_HIGH_FREQUENCY_TAIL:
        signed_kx = np.sign(sideband_sign) * kx_axis
        lower_edge = bc - bh
        mask = signed_kx > lower_edge
        if taper > 0:
            flat = signed_kx >= lower_edge + taper
            roll = (signed_kx > lower_edge) & (signed_kx < lower_edge + taper)
            weights[flat] = 1.0
            weights[roll] = 0.5 * (
                1.0 - np.cos(np.pi * (signed_kx[roll] - lower_edge) / taper)
            )
        else:
            weights[mask] = 1.0
    else:
        dist = np.abs(kx_axis - center)
        mask = dist < bh
        if taper > 0:
            flat = dist <= (bh - taper)
            roll = (dist > (bh - taper)) & (dist < bh)
            weights[flat] = 1.0
            weights[roll] = 0.5 * (1.0 + np.cos(np.pi * (dist[roll] - (bh - taper)) / taper))
        else:
            weights[mask] = 1.0
     
    Ff = F * weights[None, :]
    return Ff, mask, conjugate_mask


def plot_filtered_spatial_fft(F, kx_axis, tilt_angle, title_suffix=""):
    """Figure 4: Filtered spatial FFT."""
    ck = 2 * np.pi * np.sin(tilt_angle) / CENTER_WAVELENGTH
    bc = abs(ck)
    dh = bc * 0.15
    Ff, mp, mn = filter_spatial_sideband(F, kx_axis, tilt_angle)
    
    l_ax = np.linspace(LAMBDA_START, LAMBDA_END, N_SPECTRAL) * 1e9
    kxu = kx_axis / 1e6
    
    Ffm = np.abs(Ff)
    Ffd = np.zeros_like(Ffm)
    for i in range(N_SPECTRAL):
        r = Ffm[i, :]
        if r.max() > 0: Ffd[i, :] = 20 * np.log10(r / r.max() + 1e-10)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    axs[0,0].imshow(np.tile(mp.astype(float), (N_SPECTRAL, 1)),
                    aspect='auto', cmap='gray',
                    extent=[kxu[0], kxu[-1], l_ax[0], l_ax[-1]], origin='lower')
    axs[0,0].set_xlabel('kx [rad/μm]'); axs[0,0].set_ylabel('λ [nm]')
    axs[0,0].set_title('Filter mask')
    
    Fod = np.zeros_like(np.abs(F))
    for i in range(N_SPECTRAL):
        r = np.abs(F[i, :])
        if r.max() > 0: Fod[i, :] = 20 * np.log10(r / r.max() + 1e-10)
    axs[0,1].imshow(Fod, aspect='auto', vmin=-30, vmax=0,
                    extent=[kxu[0], kxu[-1], l_ax[0], l_ax[-1]],
                    origin='lower', cmap='hot', alpha=0.7)
    if np.any(mp):
        axs[0,1].axvspan(kxu[mp][0], kxu[mp][-1], color='lime', alpha=0.15, label='Keep')
    if np.any(mn):
        axs[0,1].axvspan(kxu[mn][0], kxu[mn][-1], color='red', alpha=0.1, label='Remove')
    axs[0,1].axvspan(-dh/1e6, dh/1e6, color='gray', alpha=0.2, label='DC')
    axs[0,1].set_xlabel('kx [rad/μm]'); axs[0,1].set_ylabel('λ [nm]')
    axs[0,1].set_title('Original + filter'); axs[0,1].legend(fontsize=8)
    
    im = axs[1,0].imshow(Ffd, aspect='auto', vmin=-30, vmax=0,
                          extent=[kxu[0], kxu[-1], l_ax[0], l_ax[-1]],
                          origin='lower', cmap='hot')
    axs[1,0].set_xlabel('kx [rad/μm]'); axs[1,0].set_ylabel('λ [nm]')
    axs[1,0].set_title(f'After filter {title_suffix}')
    plt.colorbar(im, ax=axs[1,0], fraction=0.046)
    
    ko = np.abs(F).sum(axis=0)
    kf = np.abs(Ff).sum(axis=0)
    axs[1,1].plot(kxu, ko/ko.max(), 'b-', lw=0.8, label='Original')
    axs[1,1].plot(kxu, kf/kf.max(), 'r-', lw=0.8, label='Filtered')
    axs[1,1].set_xlabel('kx [rad/μm]'); axs[1,1].set_ylabel('Normalized')
    axs[1,1].set_title('Integrated kx'); axs[1,1].legend(fontsize=8)
    axs[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.show()
    return fig, Ff


def reconstruct_from_filtered(F_filtered, x_cam, tilt_angle,
                              sideband_sign=SIDEBAND_SIGN,
                              remove_tilt=True):
    """IFFT along X + reference-carrier demodulation + FFT along k."""
    I_f = np.fft.ifft(np.fft.ifftshift(F_filtered, axes=1), axis=1)
    if remove_tilt and abs(tilt_angle) > 1e-12:
        carrier = np.exp(
            -1j * sideband_sign
            * k_array[:, None]
            * x_cam[None, :]
            * np.sin(tilt_angle)
        )
        I_f = I_f * carrier
        if sideband_sign > 0:
            # The positive kx sideband is R * conj(E_sample).  Conjugate it
            # after carrier removal so the axial phase has the sample sign.
            I_f = np.conj(I_f)
    if SUBTRACT_COMMON_MODE_SPECTRUM:
        I_f = I_f - np.median(I_f, axis=1, keepdims=True)
    if SPECTRAL_WINDOW is not None:
        I_f = I_f * spectral_window_vector(I_f.shape[0])[:, None]
    XZ = np.fft.fftshift(np.fft.fft(I_f, axis=0), axes=0)
    z_max = Z_MAX
    return np.abs(XZ).T, z_max, I_f


def plot_filtered_interference(I_f, x_cam, title_suffix=""):
    """Figure 5: Filtered analytic signal (Re, |I|, Phase)."""
    l_ax = np.linspace(LAMBDA_START, LAMBDA_END, N_SPECTRAL) * 1e9
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    
    axs[0].imshow(np.real(I_f), aspect='auto', cmap='RdBu',
                  extent=[l_ax[0], l_ax[-1], x_cam[0]*1e3, x_cam[-1]*1e3], origin='lower')
    axs[0].set_xlabel('λ [nm]'); axs[0].set_ylabel('Camera X [mm]')
    axs[0].set_title(f'Re(I_filtered) {title_suffix}')
    
    axs[1].imshow(np.abs(I_f), aspect='auto', cmap='hot',
                  extent=[l_ax[0], l_ax[-1], x_cam[0]*1e3, x_cam[-1]*1e3], origin='lower')
    axs[1].set_xlabel('λ [nm]'); axs[1].set_ylabel('Camera X [mm]')
    axs[1].set_title(f'|I_filtered| {title_suffix}')
    
    axs[2].imshow(np.angle(I_f), aspect='auto', cmap='twilight', vmin=-np.pi, vmax=np.pi,
                  extent=[l_ax[0], l_ax[-1], x_cam[0]*1e3, x_cam[-1]*1e3], origin='lower')
    axs[2].set_xlabel('λ [nm]'); axs[2].set_ylabel('Camera X [mm]')
    axs[2].set_title(f'Phase(I_filtered) {title_suffix}')
    
    plt.tight_layout(); plt.show()
    return fig


def spectral_window_vector(n, window_type=SPECTRAL_WINDOW):
    """Return a normalized spectral apodization window."""
    if window_type is None or str(window_type).lower() in ("none", "off", ""):
        return np.ones(n, dtype=np.float32)

    wt = str(window_type).lower()
    if wt == "hann":
        w = np.hanning(n)
    elif wt == "hamming":
        w = np.hamming(n)
    elif wt == "blackman":
        w = np.blackman(n)
    else:
        raise ValueError(f"Unknown SPECTRAL_WINDOW: {window_type}")

    w = w.astype(np.float32)
    return w / max(float(np.mean(w)), 1e-12)


def display_norm_value(XZ, z_ax, dc_exclude_mm=DISPLAY_DC_EXCLUDE_MM):
    """Robust display normalization that ignores the residual DC region."""
    keep = np.abs(z_ax) >= dc_exclude_mm
    if not np.any(keep):
        keep = np.ones_like(z_ax, dtype=bool)
    vmax = np.percentile(XZ[:, keep], 99.8)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.max(XZ[:, keep])
    return max(float(vmax), 1e-12)


def plot_full_depth_xz(XZ_full, XZ_1d, title_prefix=""):
    """Figure 6: XZ comparison."""
    z_ax = np.linspace(-Z_MAX*1e3, Z_MAX*1e3, N_SPECTRAL)
    x_ax = np.linspace(-CAMERA_FOV_X/2, CAMERA_FOV_X/2, N_X) / M * 1e3
    
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    
    X1d = XZ_1d; Xf = XZ_full
    norm_1d = display_norm_value(X1d, z_ax)
    norm_full = display_norm_value(Xf, z_ax)
    X1d_dB = 20 * np.log10(X1d / norm_1d + 1e-10)
    
    axs[0,0].imshow(X1d_dB, aspect='auto', vmin=-40, vmax=0,
                    extent=[z_ax[0], z_ax[-1], x_ax[0], x_ax[-1]],
                    origin='lower', cmap='hot')
    axs[0,0].set_xlabel('Z [mm]'); axs[0,0].set_ylabel('X [mm]')
    axs[0,0].set_title(f'{title_prefix}Without Tilt Filtering (dB)')
    
    axs[0,1].imshow(np.clip(X1d/norm_1d, 0, 1), aspect='auto',
                    extent=[z_ax[0], z_ax[-1], x_ax[0], x_ax[-1]],
                    origin='lower', cmap='hot')
    axs[0,1].set_xlabel('Z [mm]'); axs[0,1].set_ylabel('X [mm]')
    axs[0,1].set_title('Without Tilt Filtering (linear)')
    
    ci = N_X // 2
    axs[0,2].plot(z_ax, X1d[ci,:]/norm_1d, 'b-', lw=0.8)
    axs[0,2].set_xlabel('Z [mm]'); axs[0,2].set_ylabel('Intensity')
    axs[0,2].set_title('Center A-scan: Without Tilt Filtering'); axs[0,2].grid(True, alpha=0.3)
    
    Xf_dB = 20 * np.log10(Xf / norm_full + 1e-10)
    axs[1,0].imshow(Xf_dB, aspect='auto', vmin=-40, vmax=0,
                    extent=[z_ax[0], z_ax[-1], x_ax[0], x_ax[-1]],
                    origin='lower', cmap='hot')
    axs[1,0].set_xlabel('Z [mm]'); axs[1,0].set_ylabel('X [mm]')
    axs[1,0].set_title(f'{title_prefix}With Tilt + Sideband Filtering (dB)')
    
    axs[1,1].imshow(np.clip(Xf/norm_full, 0, 1), aspect='auto',
                    extent=[z_ax[0], z_ax[-1], x_ax[0], x_ax[-1]],
                    origin='lower', cmap='hot')
    axs[1,1].set_xlabel('Z [mm]'); axs[1,1].set_ylabel('X [mm]')
    axs[1,1].set_title('With Tilt + Sideband Filtering (linear)')
    
    axs[1,2].plot(z_ax, Xf[ci,:]/norm_full, 'r-', lw=0.8, label='With tilt + filtering')
    axs[1,2].plot(z_ax, X1d[ci,:]/norm_1d, 'b--', lw=0.8, alpha=0.5, label='Without filtering')
    axs[1,2].set_xlabel('Z [mm]'); axs[1,2].set_ylabel('Intensity')
    axs[1,2].set_title('Center A-scan Comparison'); axs[1,2].legend(fontsize=8)
    axs[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.show()
    return fig


# ============================================================================
# CACHE
# ============================================================================
CACHE_DIR = "cache"
PHANTOM_FILE  = os.path.join(CACHE_DIR, "phantom.npz")
CACHE_FILE_OFF = os.path.join(CACHE_DIR, "interference_off.npz")
CACHE_FILE_ON  = os.path.join(CACHE_DIR, "interference_on.npz")
HASH_FILE     = os.path.join(CACHE_DIR, "params_hash.json")


def compute_params_hash():
    d = {
        "CENTER_WAVELENGTH": float(CENTER_WAVELENGTH),
        "WAVELENGTH_RANGE": float(WAVELENGTH_RANGE),
        "PHANTOM_VERSION": int(PHANTOM_VERSION),
        "SAMPLE_FOV_X": float(SAMPLE_FOV_X),
        "NA": float(NA),
        "N_SPECTRAL": int(N_SPECTRAL),
        "N_X": int(N_X),
        "PIXEL_SIZE": float(PIXEL_SIZE),
        "REF_TILT_ANGLE_DEG": float(REF_TILT_ANGLE_DEG),
        "REF_AMP": float(REF_AMP),
        "SAMPLE_FIELD_SCALE": float(SAMPLE_FIELD_SCALE),
        "INCLUDE_SAMPLE_SELF_INTERFERENCE": bool(INCLUDE_SAMPLE_SELF_INTERFERENCE),
        "REFERENCE_Z": float(REFERENCE_Z),
        "FOCUS_Z": float(FOCUS_Z),
        "SCATTER_X_RANGE": float(SCATTER_X_RANGE),
        "SCATTER_Z_RANGE": float(SCATTER_Z_RANGE),
        "SCATTER_MIN_REFL": float(SCATTER_MIN_REFL),
        "SCATTER_MAX_REFL": float(SCATTER_MAX_REFL),
        "FULL_WELL_DEPTH": int(FULL_WELL_DEPTH),
        "WELL_FILL_FRACTION": float(WELL_FILL_FRACTION),
        "READOUT_NOISE_E": float(READOUT_NOISE_E),
    }
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def check_cache_valid():
    if not os.path.exists(HASH_FILE): return False
    try:
        with open(HASH_FILE, "r") as f:
            if json.load(f).get("hash") != compute_params_hash(): return False
        return all(os.path.exists(p) for p in [PHANTOM_FILE, CACHE_FILE_OFF, CACHE_FILE_ON])
    except: return False


def save_cache(xs, zs, a, Io, Ion, xc):
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(PHANTOM_FILE, xs=xs, zs=zs, a=a, xc=xc)
    np.savez_compressed(CACHE_FILE_OFF, I=Io)
    np.savez_compressed(CACHE_FILE_ON, I=Ion)
    with open(HASH_FILE, "w") as f:
        json.dump({"hash": compute_params_hash()}, f)
    print(f"  Cached to {CACHE_DIR}/")


def load_cache():
    try:
        ph = np.load(PHANTOM_FILE)
        return (ph["xs"], ph["zs"], ph["a"],
                np.load(CACHE_FILE_OFF)["I"],
                np.load(CACHE_FILE_ON)["I"], ph["xc"])
    except: return None


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 67)
    print("OFF-AXIS FF-OCT SIMULATION")
    print("=" * 67)
    
    print("\n--- Cache check ---")
    cached = load_cache() if check_cache_valid() else None
    
    if cached is not None:
        xs, zs, a, I_off, I_on, x_cam = cached
        print(f"  Loaded: {len(xs)} scatterers, {I_off.shape} interference")
    else:
        print("\n--- Step 1: Generating eye phantom ---")
        xs, zs, a = generate_eye_phantom()
        
        print("\n--- Step 2: Generating interference (GPU) ---")
        I_off, x_cam = generate_interference(xs, zs, a, REF_TILT_ANGLE)
        I_on,  _      = generate_interference(xs, zs, a, 0.0)
        print(f"  Off: {I_off.shape}, On: {I_on.shape}")
        
        print("\n--- Saving to cache ---")
        save_cache(xs, zs, a, I_off, I_on, x_cam)
    
    print("\n--- Adding camera noise ---")
    I_off_n, ni = add_camera_noise(I_off, MAX_SIGNAL_E, READOUT_NOISE_E)
    I_on_n,  _  = add_camera_noise(I_on, MAX_SIGNAL_E, READOUT_NOISE_E, rng_seed=456)
    print(f"  SNR: {ni['SNR_dB']:.1f} dB")
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ts = time.strftime("%Y%m%d_%H%M%S")
    print(f"\n--- Saving figures to {SCRIPT_DIR}/ ---")
    
    # Fig 1: Phantom
    plot_phantom(xs, zs, a).savefig(os.path.join(SCRIPT_DIR, f"01_phantom_{ts}.png"), dpi=150, bbox_inches='tight')
    plot_ground_truth_xz(xs, zs, a).savefig(
        os.path.join(SCRIPT_DIR, f"01_ground_truth_xz_{ts}.png"), dpi=150, bbox_inches='tight')
    
    # Fig 2: Interference
    plot_interference(I_off_n, x_cam, "(Off-axis, θ=2°, noisy)", ni).savefig(
        os.path.join(SCRIPT_DIR, f"02_interference_{ts}.png"), dpi=150, bbox_inches='tight')
    
    # Fig 3: Spatial FFT
    fig3, F_off, kx = plot_spatial_fft(I_off_n, REF_TILT_ANGLE, "(Off-axis)")
    fig3.savefig(os.path.join(SCRIPT_DIR, f"03_spatial_fft_off_{ts}.png"), dpi=150, bbox_inches='tight')
    fig3_on, _, _ = plot_spatial_fft(I_on_n, 0.0, "(On-axis)")
    fig3_on.savefig(os.path.join(SCRIPT_DIR, f"03_spatial_fft_on_{ts}.png"), dpi=150, bbox_inches='tight')
    
    # Fig 4: Bandpass filter
    fig4, F_filt = plot_filtered_spatial_fft(F_off, kx, REF_TILT_ANGLE, "(Off-axis)")
    fig4.savefig(os.path.join(SCRIPT_DIR, f"04_filtered_fft_{ts}.png"), dpi=150, bbox_inches='tight')
    
    # Step 5: Reconstruct
    XZ_full, _, I_filt = reconstruct_from_filtered(F_filt, x_cam, REF_TILT_ANGLE)
    spectral_win = spectral_window_vector(N_SPECTRAL)[:, None]
    I_on_for_fft = I_on_n.astype(np.complex64) * spectral_win
    XZ_1d_on = np.abs(np.fft.fftshift(np.fft.fft(I_on_for_fft, axis=0), axes=0)).T
    
    # Fig 5: Filtered analytic signal
    plot_filtered_interference(I_filt, x_cam, "(After filtering)").savefig(
        os.path.join(SCRIPT_DIR, f"05_filtered_analytic_{ts}.png"), dpi=150, bbox_inches='tight')
    
    # Fig 6: XZ comparison
    plot_full_depth_xz(XZ_full, XZ_1d_on, "").savefig(
        os.path.join(SCRIPT_DIR, f"06_xz_comparison_{ts}.png"), dpi=150, bbox_inches='tight')
    
    # Fig 7: On vs Off
    z_ax = np.linspace(-Z_MAX*1e3, Z_MAX*1e3, N_SPECTRAL)
    x_ax = np.linspace(-CAMERA_FOV_X/2, CAMERA_FOV_X/2, N_X) / M * 1e3
    
    fig7, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    norm_on = display_norm_value(XZ_1d_on, z_ax)
    norm_off = display_norm_value(XZ_full, z_ax)
    dBo = 20 * np.log10(XZ_1d_on / norm_on + 1e-10)
    axs[0].imshow(dBo, aspect='auto', vmin=-40, vmax=0,
                  extent=[z_ax[0], z_ax[-1], x_ax[0], x_ax[-1]], origin='lower', cmap='hot')
    axs[0].set_xlabel('Z [mm]'); axs[0].set_ylabel('X [mm]')
    axs[0].set_title('Without Tilt Processing')
    
    dBf = 20 * np.log10(XZ_full / norm_off + 1e-10)
    axs[1].imshow(dBf, aspect='auto', vmin=-40, vmax=0,
                  extent=[z_ax[0], z_ax[-1], x_ax[0], x_ax[-1]], origin='lower', cmap='hot')
    axs[1].set_xlabel('Z [mm]'); axs[1].set_ylabel('X [mm]')
    axs[1].set_title('With Tilt + Sideband Filtering')
    
    plt.tight_layout()
    fig7.savefig(os.path.join(SCRIPT_DIR, f"07_on_vs_off_{ts}.png"), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 67)
    print("SIMULATION COMPLETE")
    print("=" * 67)
    print(f"  SNR: {ni['SNR_dB']:.1f} dB")
    print("\n  Figures: 1-Phantom/Ground truth, 2-Interference, 3-FFT(off+on), 4-Filter, 5-Analytic, 6-XZ, 7-Comparison")
