"""
driver_gw.py
============
Orchestration layer for GravitationalWaveSources.

Calls integrate_inspiral() → plot_inspiral() → prints summary table.
"""

import physics_gw as phys
import plot_gw    as viz


def run(m1_msun=1.4, m2_msun=1.4, d_mpc=400.0,
        dt=1e-4, f_start=10.0,
        outdir=None,
        t_before=None, t_after=None, lw=0.4,
        n_ringdown_tau=6, ringdown_pts=4000,
        dpi=150):
    """
    Full pipeline: integrate → plot → summary.

    Parameters
    ----------
    m1_msun, m2_msun : component masses [M_sun]
    d_mpc            : luminosity distance [Mpc]
    dt               : RK4 time step [s]
    f_start          : initial GW frequency [Hz]
    outdir           : output directory for PNG (None → screen display)
    t_before         : seconds before merger to show (None → all)
    t_after          : seconds after  merger to show (None → all)
    lw               : waveform line width [pt]
    n_ringdown_tau   : ringdown duration in units of tau_QNM
    ringdown_pts     : number of ringdown sample points
    dpi              : PNG resolution
    """
    print("[driver_gw] Integrating inspiral …")
    result = phys.integrate_inspiral(
        m1_msun, m2_msun, d_mpc,
        dt=dt, f_start=f_start,
        n_ringdown_tau=n_ringdown_tau,
        ringdown_pts=ringdown_pts,
    )

    _print_summary(result["summary"])

    print("[driver_gw] Rendering figure …")
    viz.plot_inspiral(
        result,
        outdir=outdir,
        t_before=t_before,
        t_after=t_after,
        lw=lw,
        dpi=dpi,
    )

    return result


def _print_summary(s):
    W = 54
    sep = "─" * W
    print(sep)
    print("  GravitationalWaveSources — Run Summary")
    print(sep)
    print(f"  Masses           : {s['m1_msun']:.3f} + {s['m2_msun']:.3f}  M☉")
    print(f"  Total mass       : {s['M_total_msun']:.3f}  M☉")
    print(f"  Chirp mass       : {s['Mc_msun']:.4f}  M☉")
    print(f"  Distance         : {s['d_mpc']:.1f}  Mpc")
    print(f"  Band start       : {s['f_start_hz']:.1f}  Hz")
    print(f"  ISCO frequency   : {s['f_isco_hz']:.1f}  Hz")
    print(f"  Time in band     : {s['T_band_s']:.2f}  s")
    print(f"  Peak strain      : {s['A_peak']:.3e}")
    print(f"  Final mass (est) : {s['M_final_msun']:.3f}  M☉")
    print(f"  QNM frequency    : {s['f_qnm_hz']:.1f}  Hz")
    print(f"  QNM decay time   : {s['tau_qnm_ms']:.3f}  ms")
    print(sep)
