"""
physics_gw.py
=============
Core physics engine for the GravitationalWaveSources program.

Computes the leading-order Peters/quadrupole inspiral for a quasi-circular
binary, then appends a Schwarzschild quasi-normal-mode (QNM) ringdown after
the ISCO crossing.

SI units throughout; results returned in SI.
"""

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
G   = 6.674_30e-11   # m³ kg⁻¹ s⁻²
c   = 2.997_924_58e8 # m s⁻¹
M_sun = 1.988_92e30  # kg


# ── Helper: chirp mass ────────────────────────────────────────────────────────
def chirp_mass(m1_kg, m2_kg):
    """Return chirp mass in kg."""
    return (m1_kg * m2_kg) ** 0.6 / (m1_kg + m2_kg) ** 0.2


# ── Inspiral ODE (df/dt) ──────────────────────────────────────────────────────
def dfdt(f, Mc_kg):
    """
    Time derivative of GW frequency under Peters quadrupole formula.
    f   : GW frequency [Hz]
    Mc  : chirp mass   [kg]
    """
    Mc_geom = G * Mc_kg / c**3          # chirp mass in seconds
    return (96.0 / 5.0) * np.pi**(8.0/3.0) * Mc_geom**(5.0/3.0) * f**(11.0/3.0)


# ── Strain amplitude ──────────────────────────────────────────────────────────
def strain_amplitude(f_hz, Mc_kg, d_m):
    """
    Leading-order dimensionless strain amplitude (sky-and-polarisation averaged).
    f   : GW frequency [Hz]
    Mc  : chirp mass   [kg]
    d   : luminosity distance [m]
    """
    Mc_geom = G * Mc_kg / c**3          # [s]
    return (4.0 * G * Mc_kg / (c**2 * d_m)) * (np.pi * G * Mc_kg * f_hz / c**3) ** (2.0/3.0)


# ── ISCO frequency ────────────────────────────────────────────────────────────
def f_isco(M_total_kg):
    """GW frequency at the Schwarzschild ISCO (6 GM/c²)."""
    return c**3 / (6.0**1.5 * np.pi * G * M_total_kg)


# ── QNM parameters (Schwarzschild, dominant l=m=2 mode) ─────────────────────
def qnm_params(M_final_kg):
    """
    Return (f_qnm [Hz], tau_qnm [s]) for the dominant Schwarzschild QNM.

    Dimensionless frequencies from Leaver (1985) / Berti et al. (2009):
        omega_r  = 0.3737  (real part, in units of c³/GM_f)
        omega_i  = 0.0890  (imaginary part)

    f_qnm  = omega_r * c³ / (2π G M_f)
    tau    = 1 / (omega_i * c³ / (G M_f))   [= G M_f / (omega_i c³)]
    """
    scale = G * M_final_kg / c**3          # [s]  -- GM_f/c³
    f_qnm  = 0.3737 / (2.0 * np.pi * scale)
    tau    = scale / 0.0890
    return f_qnm, tau


# ── Main integration routine ──────────────────────────────────────────────────
def integrate_inspiral(m1_msun, m2_msun, d_mpc,
                       dt=1e-4, f_start=10.0,
                       n_ringdown_tau=6, ringdown_pts=4000):
    """
    Integrate the quasi-circular inspiral from f_start Hz up to f_ISCO,
    then append a QNM ringdown.

    Parameters
    ----------
    m1_msun, m2_msun : component masses [solar masses]
    d_mpc            : luminosity distance [Mpc]
    dt               : time step for inspiral integration [s]
    f_start          : initial GW frequency [Hz]
    n_ringdown_tau   : ringdown duration in units of tau_qnm
    ringdown_pts     : number of sample points in the ringdown segment

    Returns
    -------
    dict with keys:
        t        : time array [s], t=0 at start, merger at t_merger
        h        : dimensionless strain h(t)
        A        : strain envelope A(t)  (inspiral portion only; NaN in ringdown)
        f        : instantaneous GW frequency [Hz] (inspiral only; NaN in ringdown)
        t_merger : time of ISCO crossing [s]
        f_isco_hz: ISCO frequency [Hz]
        Mc_msun  : chirp mass [solar masses]
        summary  : dict of scalar diagnostics
    """
    m1 = m1_msun * M_sun
    m2 = m2_msun * M_sun
    M_total = m1 + m2
    Mc = chirp_mass(m1, m2)
    d  = d_mpc * 3.085_677_581_49e22   # Mpc → m

    f_isco_hz = f_isco(M_total)

    # ── RK4 integration of df/dt ─────────────────────────────────────────────
    t_list, f_list, A_list, h_list = [], [], [], []
    t = 0.0
    f = float(f_start)
    phase = 0.0                         # accumulated GW phase [rad]

    while f < f_isco_hz:
        A = strain_amplitude(f, Mc, d)
        h = A * np.cos(phase)
        t_list.append(t)
        f_list.append(f)
        A_list.append(A)
        h_list.append(h)

        # RK4 on f
        k1 = dfdt(f, Mc)
        k2 = dfdt(f + 0.5*dt*k1,   Mc)
        k3 = dfdt(f + 0.5*dt*k2,   Mc)
        k4 = dfdt(f +     dt*k3,   Mc)
        df = (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        phase += 2.0 * np.pi * f * dt
        f  += df
        t  += dt

    # Record merger time
    t_merger = t

    # Peak strain at merger (used to normalise ringdown)
    A_peak = strain_amplitude(f_isco_hz, Mc, d)

    # ── QNM Ringdown ─────────────────────────────────────────────────────────
    # Approximate final mass: ~95 % of total (radiated ~5 % in GW)
    M_final = 0.95 * M_total
    f_qnm, tau_qnm = qnm_params(M_final)

    t_rd_end = n_ringdown_tau * tau_qnm
    t_rd = np.linspace(0.0, t_rd_end, ringdown_pts)
    h_rd = A_peak * np.cos(2.0 * np.pi * f_qnm * t_rd) * np.exp(-t_rd / tau_qnm)

    # NaN sentinels for envelope / frequency during ringdown
    A_rd = np.full_like(t_rd, np.nan)
    f_rd = np.full_like(t_rd, np.nan)

    # ── Concatenate ──────────────────────────────────────────────────────────
    t_arr = np.concatenate([np.array(t_list), t_merger + t_rd])
    h_arr = np.concatenate([np.array(h_list), h_rd])
    A_arr = np.concatenate([np.array(A_list), A_rd])
    f_arr = np.concatenate([np.array(f_list), f_rd])

    # ── Time-to-merger (re-zero so merger is at t=0) ─────────────────────────
    # Leave t as absolute (t=0 at start of observation) so the user can
    # see the full inspiral duration; t_merger marks the boundary.

    # ── Scalar diagnostics ───────────────────────────────────────────────────
    T_band = t_merger                   # inspiral duration in band [s]
    summary = dict(
        m1_msun    = m1_msun,
        m2_msun    = m2_msun,
        Mc_msun    = Mc / M_sun,
        M_total_msun = M_total / M_sun,
        d_mpc      = d_mpc,
        f_start_hz = f_start,
        f_isco_hz  = f_isco_hz,
        f_qnm_hz   = f_qnm,
        tau_qnm_ms = tau_qnm * 1e3,
        t_merger_s = t_merger,
        T_band_s   = T_band,
        A_peak     = A_peak,
        M_final_msun = M_final / M_sun,
    )

    return dict(
        t        = t_arr,
        h        = h_arr,
        A        = A_arr,
        f        = f_arr,
        t_merger = t_merger,
        f_isco_hz= f_isco_hz,
        Mc_msun  = Mc / M_sun,
        summary  = summary,
    )
