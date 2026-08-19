"""
driver_sev.py
=============
Orchestration layer for StellarEvolutionTracks.

The driver validates the presentation-level inputs, dispatches to the
requested physics calculation, prints a run summary, writes an optional
CSV data file, and hands the result to the plotting layer.
"""

import csv
import math
import os
from datetime import datetime

import numpy as np

import physics_sev as phys
import plot_sev as viz

MODES = ("tracks", "hr", "wdcool", "nsmr")

W = 68
SEP = "-" * W


# ======================================================================
# Validation helpers
# ======================================================================
def _finite(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _validate_output(outdir, csvdir, dpi, lw):
    dpi = _finite("dpi", dpi)
    lw = _finite("lw", lw)
    if int(dpi) != dpi or dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    if not (10 <= dpi <= 1200):
        raise ValueError("dpi must lie between 10 and 1200.")
    if lw <= 0:
        raise ValueError("lw must be greater than zero.")
    for name, path in (("outdir", outdir), ("csvdir", csvdir)):
        if path is None:
            continue
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{name} must be a non-empty directory path.")
        if os.path.exists(path) and not os.path.isdir(path):
            raise ValueError(f"{name} = {path!r} exists but is not a directory.")
    return int(dpi), lw


def _parse_float_list(name, text, lo=None, hi=None, max_items=None):
    """Parse a comma-separated list of numbers supplied on the command line."""
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        items = list(text)
    else:
        items = [piece for piece in str(text).replace(";", ",").split(",")
                 if piece.strip()]
    if not items:
        raise ValueError(f"{name} did not contain any numbers.")
    values = []
    for piece in items:
        try:
            v = float(piece)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} contains {piece!r}, which is not a number."
            ) from exc
        if not math.isfinite(v):
            raise ValueError(f"{name} contains a non-finite value.")
        if lo is not None and v < lo:
            raise ValueError(f"{name} contains {v:g}, which is below {lo:g}.")
        if hi is not None and v > hi:
            raise ValueError(f"{name} contains {v:g}, which is above {hi:g}.")
        values.append(v)
    if max_items is not None and len(values) > max_items:
        raise ValueError(f"{name} may contain at most {max_items} entries.")
    return values


# ======================================================================
# CSV output
# ======================================================================
def _write_csv(csvdir, prefix, header, rows, comments=()):
    os.makedirs(csvdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(csvdir, f"{prefix}_{stamp}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        for line in comments:
            fh.write(f"# {line}\n")
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[driver_sev] CSV saved -> {path}")
    return path


def _track_rows(result):
    s = result["summary"]
    names = {0: "MS", 1: "SGB", 2: "RGB"}
    rows = []
    for i in range(result["t"].size):
        rows.append([
            f"{s['m_msun']:.4f}",
            f"{result['t'][i] / phys.GYR:.8g}",
            names[int(result["phase"][i])],
            f"{result['Xc'][i]:.6f}",
            f"{result['Mcore'][i]:.6f}",
            f"{result['mu'][i]:.6f}",
            f"{result['L'][i]:.6g}",
            f"{result['R'][i]:.6g}",
            f"{result['Teff'][i]:.6g}",
            f"{np.log10(result['Teff'][i]):.6f}",
            f"{np.log10(result['L'][i]):.6f}",
        ])
    return rows


TRACK_HEADER = ["M_Msun", "age_Gyr", "phase", "X_c", "Mcore_Msun", "mu_eff",
                "L_Lsun", "R_Rsun", "Teff_K", "log10_Teff", "log10_L"]


# ======================================================================
# Summary printers
# ======================================================================
def _head(title):
    print(SEP)
    print(f"  StellarEvolutionTracks — {title}")
    print(SEP)


def _print_track_summary(s):
    _head("single-star evolution")
    print(f"  Mass                : {s['m_msun']:.3f}  Msun")
    print(f"  Composition         : X = {s['X']:.3f},  Y = {s['Y']:.3f},  Z = {s['Z']:.4f}")
    print(f"  Envelope mu         : {s['mu_env']:.4f}")
    print(f"  Burning / opacity   : {s['burning']} (nu = {s['nu']:.0f}) / {s['opacity']}"
          f"  (a = {s['kappa_a']:.1f}, b = {s['kappa_b']:.1f})")
    print(f"  Homology exponents  : L ~ mu^{s['e_L_mu']:.3f} M^{s['e_L_M']:.3f},"
          f"   R ~ mu^{s['p_R_mu']:.3f} M^{s['q_R_M']:.3f}")
    print(f"  Core fraction q_c   : {s['qc']:.4f}"
          f"   (core weight w = {s['core_weight']:.3f},"
          f" expansion = {s['expansion']:.2f})")
    print(SEP)
    print(f"  ZAMS                : L = {s['L_zams']:.4g} Lsun,"
          f"  R = {s['R_zams']:.4g} Rsun,  Teff = {s['T_zams']:.0f} K")
    if s["truncated"]:
        print(f"  Main sequence       : still burning at t_max = {s['t_max_gyr']:g} Gyr")
        print(f"  Central X_c reached : {s['X']:.3f} -> see the data file")
    else:
        print(f"  TAMS                : L = {s['L_tams']:.4g} Lsun,"
              f"  Teff = {s['T_tams']:.0f} K")
        print(f"  MS lifetime         : {s['t_ms_gyr']:.4g}  Gyr")
    print(f"  Helium core at TAMS : {s['mc_tams']:.4f}  Msun"
          f"   (core efficiency {s['core_efficiency']:.2f})")
    if s["post_ms"]:
        print(f"  Post-MS regime      : {s['post_regime']}")
        print(f"  Post-MS duration    : {s['t_post_gyr']:.4g}  Gyr")
        print(f"  Ends at             : {s['phase_end']}")
        print(f"  Core mass at end    : {s['mc_ign']:.4f}  Msun")
        print(f"  Luminosity at end   : {s['L_tip']:.5g}  Lsun")
        print(f"  Total age at end    : {s['t_total_gyr']:.4g}  Gyr")
    else:
        print("  Post-MS             : not computed")
    print(SEP)
    print(f"  Predicted remnant   : {s['remnant_kind']}"
          f",  about {s['remnant_msun']:.3f} Msun")
    print(f"                        ({s['remnant_note']})")
    print(f"  Track points        : {s['n_points']:,}")
    print(SEP)


def _print_hr_summary(s):
    _head("HR-diagram track grid")
    print(f"  Composition         : X = {s['X']:.3f},  Z = {s['Z']:.4f}")
    print(f"  Tracks              : {s['n_tracks']}")
    print(f"  Isochrones          : {s['n_isochrones']}"
          + (f"  at {', '.join(f'{a:g}' for a in s['isochrone_ages'])} Gyr"
             if s["n_isochrones"] else ""))
    print(SEP)
    print(f"  {'M [Msun]':>10}  {'t_MS [Gyr]':>12}  {'t_end [Gyr]':>12}")
    for m, tms, tot in zip(s["masses"], s["lifetimes_gyr"], s["totals_gyr"]):
        print(f"  {m:10.3f}  {tms:12.4g}  {tot:12.4g}")
    print(SEP)


def _print_wd_summary(s):
    _head("white-dwarf structure and cooling")
    print(f"  Mass                : {s['m_msun']:.4f}  Msun"
          f"   (requested {s['m_requested']:.4f})")
    print(f"  mu_e (core)         : {s['mu_e']:.3f}"
          f"   -> Chandrasekhar limit {s['M_ch']:.4f} Msun")
    print(f"  Central density     : {s['rho_c']:.4e}  kg/m^3")
    print(f"  Radius              : {s['R_km']:.1f} km"
          f"  = {s['R_rearth']:.3f} Earth radii = {s['R_rsun']:.5f} Rsun")
    print(f"  Mean density        : {s['mean_density']:.4e}  kg/m^3")
    print(f"  Surface gravity     : {s['surface_gravity']:.4e}  m/s^2")
    print(SEP)
    print(f"  Envelope            : X = {s['X_env']:.3f}, Z = {s['Z_env']:g},"
          f"  mu = {s['mu_env']:.4f},  mu_e = {s['mu_e_env']:.4f}")
    print(f"  Kramers kappa_0     : {s['kappa0']:.4e}  SI")
    print(f"  Ion mass number A   : {s['A_ion']:.1f}")
    print(f"  Mestel coefficient  : {s['mestel_C']:.4e}  (L = C M T_c^7/2, SI)")
    print(SEP)
    print(f"  Core temperature    : {s['Tc0']:.3e} K  ->  {s['Tc_end']:.3e} K")
    print(f"  Luminosity          : {s['L_start']:.4e}  ->  {s['L_end']:.4e}  Lsun")
    print(f"  Effective temp.     : {s['Teff_start']:.0f} K  ->  {s['Teff_end']:.0f} K")
    print(f"  Cooling age (RK4)   : {s['t_end_gyr']:.5g}  Gyr")
    print(f"  Cooling age (exact) : {s['t_end_analytic_gyr']:.5g}  Gyr")
    rel = abs(s["t_end_gyr"] - s["t_end_analytic_gyr"]) / s["t_end_analytic_gyr"]
    print(f"  Relative difference : {rel:.3e}")
    print(f"  Cooling points      : {s['n_points']:,}")
    print(SEP)


def _print_ns_summary(s):
    _head("neutron-star mass-radius relation")
    print(f"  Equation of state   : {s['eos']}")
    if s["eos"] == "polytrope":
        print(f"  Polytropic index    : Gamma = {s['gamma']:.3f}")
        print(f"  Stiffness           : p_nuc = {s['p_nuc']:.5g}"
              f"   (K = {s['K']:.4e} SI)")
    print(f"  Gravity             : "
          + ("Tolman-Oppenheimer-Volkoff (general relativistic)"
             if s["relativistic"] else "Newtonian"))
    print(f"  Central densities   : {s['rho_lo']:.3e} .. {s['rho_hi']:.3e} kg/m^3"
          f"   ({s['n_models']} models)")
    print(SEP)
    print(f"  Maximum mass        : {s['M_max']:.4f}  Msun")
    print(f"  Radius there        : {s['R_at_Mmax']:.3f}  km")
    print(f"  Central density     : {s['rho_at_Mmax']:.4e}  kg/m^3")
    print(f"  Compactness GM/Rc^2 : {s['compact_at_Mmax']:.4f}"
          f"   (Buchdahl bound 0.4444)")
    print(f"  Surface redshift z  : {s['z_at_Mmax']:.4f}")
    print(f"  Sound speed c_s/c   : {s['cs_over_c_at_Mmax']:.4f}"
          + ("" if s["causal"] else "   *** ACAUSAL: c_s > c ***"))
    print(f"  Radius range        : {s['R_min']:.2f} .. {s['R_max']:.2f}  km")
    if not s["stable_branch"]:
        print("  NOTE: the mass is still rising at the highest central density")
        print("        sampled, so this maximum is only the end of the grid.")
        print("        Increase --rho_hi to find the true turning point.")
    print(SEP)


# ======================================================================
# Mode runners
# ======================================================================
def _run_tracks(kw, outdir, csvdir, dpi, lw):
    result = phys.integrate_track(
        m_msun=kw["mass"], X=kw["X"], Z=kw["Z"], qc=kw["qc"],
        burning=kw["burning"], opacity=kw["opacity"],
        core_weight=kw["core_weight"], expansion=kw["expansion"],
        core_efficiency=kw["core_efficiency"],
        n_ms=kw["n_ms"], n_post=kw["n_post"],
        t_max_gyr=kw["t_max"], x_end=kw["x_end"],
        include_postms=kw["postms"], homology_zams=kw["homology"],
    )
    _print_track_summary(result["summary"])
    if csvdir is not None:
        s = result["summary"]
        _write_csv(csvdir, f"sev_track_{s['m_msun']:.2f}Msun",
                   TRACK_HEADER, _track_rows(result),
                   comments=[
                       "StellarEvolutionTracks - single-star evolution track",
                       f"M = {s['m_msun']} Msun, X = {s['X']}, Z = {s['Z']}",
                       f"burning = {s['burning']}, opacity = {s['opacity']}, "
                       f"qc = {s['qc']:.4f}, core_weight = {s['core_weight']}",
                       f"t_MS = {s['t_ms_gyr']:.6g} Gyr",
                   ])
    if not kw["no_plot"]:
        viz.plot_track(result, outdir=outdir, dpi=dpi, lw=lw)
    return result


def _run_hr(kw, outdir, csvdir, dpi, lw):
    masses = _parse_float_list("masses", kw["masses"], lo=0.08, hi=120.0,
                               max_items=phys.MAX_MASSES)
    ages = _parse_float_list("isochrones", kw["isochrones"], lo=1e-6, hi=1e3,
                             max_items=10)
    result = phys.build_hr_grid(
        masses, isochrone_gyr=ages,
        X=kw["X"], Z=kw["Z"], qc=kw["qc"],
        burning=kw["burning"], opacity=kw["opacity"],
        core_weight=kw["core_weight"], expansion=kw["expansion"],
        core_efficiency=kw["core_efficiency"],
        n_ms=kw["n_ms"], n_post=kw["n_post"],
        t_max_gyr=kw["t_max"], x_end=kw["x_end"],
        include_postms=kw["postms"], homology_zams=kw["homology"],
    )
    _print_hr_summary(result["summary"])
    if csvdir is not None:
        rows = []
        for tr in result["tracks"]:
            rows.extend(_track_rows(tr))
        _write_csv(csvdir, "sev_hr_grid", TRACK_HEADER, rows,
                   comments=[
                       "StellarEvolutionTracks - HR-diagram track grid",
                       f"masses = {result['summary']['masses']}",
                       f"X = {result['summary']['X']}, Z = {result['summary']['Z']}",
                   ])
        if result["isochrones"]:
            iso_rows = []
            for iso in result["isochrones"]:
                for m, lt, ll in iso["points"]:
                    iso_rows.append([f"{iso['age_gyr']:g}", f"{m:.4f}",
                                     f"{lt:.6f}", f"{ll:.6f}"])
            _write_csv(csvdir, "sev_hr_isochrones",
                       ["age_Gyr", "M_Msun", "log10_Teff", "log10_L"], iso_rows,
                       comments=["StellarEvolutionTracks - isochrones"])
    if not kw["no_plot"]:
        viz.plot_hr_diagram(result, outdir=outdir, dpi=dpi, lw=lw)
    return result


def _run_wdcool(kw, outdir, csvdir, dpi, lw):
    result = phys.integrate_wd_cooling(
        m_msun=kw["wd_mass"], mu_e=kw["mu_e"], A_ion=kw["A_ion"],
        X_env=kw["X_env"], Z_env=kw["Z_env"],
        Tc0=kw["Tc0"], Tc_end=kw["Tc_end"],
        n_steps=kw["n_cool"], step_frac=kw["step_frac"],
    )
    _print_wd_summary(result["summary"])
    if csvdir is not None:
        s = result["summary"]
        rows = []
        for i in range(result["t"].size):
            rows.append([
                f"{result['t'][i] / phys.GYR:.8g}",
                f"{result['Tc'][i]:.6g}",
                f"{result['L'][i]:.6g}",
                f"{result['Teff'][i]:.6g}",
                f"{np.log10(result['Teff'][i]):.6f}",
                f"{np.log10(result['L'][i]):.6f}",
            ])
        _write_csv(csvdir, f"sev_wdcool_{s['m_msun']:.2f}Msun",
                   ["age_Gyr", "Tc_K", "L_Lsun", "Teff_K", "log10_Teff", "log10_L"],
                   rows,
                   comments=[
                       "StellarEvolutionTracks - white-dwarf Mestel cooling",
                       f"M = {s['m_msun']:.4f} Msun, R = {s['R_km']:.2f} km, "
                       f"rho_c = {s['rho_c']:.4e} kg/m^3",
                       f"mu_e = {s['mu_e']}, A_ion = {s['A_ion']}, "
                       f"envelope X = {s['X_env']}, Z = {s['Z_env']}",
                   ])
        mr_rows = [[f"{r:.6e}", f"{m:.6f}",
                    f"{rr * phys.R_sun / 1e3:.4f}"]
                   for r, m, rr in zip(result["mr_rho"], result["mr_M"],
                                       result["mr_R"])]
        _write_csv(csvdir, "sev_wd_mass_radius",
                   ["rho_c_kg_m3", "M_Msun", "R_km"], mr_rows,
                   comments=["StellarEvolutionTracks - cold white-dwarf "
                             "mass-radius relation"])
    if not kw["no_plot"]:
        viz.plot_wd_cooling(result, outdir=outdir, dpi=dpi, lw=lw)
    return result


def _run_nsmr(kw, outdir, csvdir, dpi, lw):
    result = phys.ns_mass_radius_curve(
        eos_name=kw["eos"], n=kw["n_mr"],
        rho_lo=kw["rho_lo"], rho_hi=kw["rho_hi"],
        relativistic=not kw["newtonian"],
        p_nuc=kw["p_nuc"], gamma=kw["gamma"],
        step_frac=kw["step_frac"],
    )
    _print_ns_summary(result["summary"])
    if csvdir is not None:
        rows = []
        for i in range(result["rho"].size):
            rows.append([
                f"{result['rho'][i]:.6e}",
                f"{result['M'][i]:.6f}",
                f"{result['R'][i]:.4f}",
                f"{result['compact'][i]:.6f}",
                f"{result['z'][i]:.6f}",
                "stable" if i <= result["i_max"] else "unstable",
            ])
        s = result["summary"]
        _write_csv(csvdir, f"sev_nsmr_{s['eos']}",
                   ["rho_c_kg_m3", "M_Msun", "R_km", "compactness_GM_Rc2",
                    "surface_redshift_z", "branch"], rows,
                   comments=[
                       "StellarEvolutionTracks - neutron-star mass-radius relation",
                       f"EOS = {s['eos']}, "
                       + ("TOV" if s["relativistic"] else "Newtonian"),
                       f"M_max = {s['M_max']:.4f} Msun at R = {s['R_at_Mmax']:.3f} km",
                   ])
    if not kw["no_plot"]:
        viz.plot_ns_mass_radius(result, outdir=outdir, dpi=dpi, lw=lw,
                                m_observed=kw["m_observed"])
    return result


# ======================================================================
# Public entry point
# ======================================================================
def run(mode="tracks",
        # --- track / HR parameters ---
        mass=1.0, masses="0.5,0.8,1.0,1.5,2.0,3.0,5.0,10.0",
        isochrones=None,
        X=0.70, Z=0.02, qc=None, burning=None, opacity="thomson",
        core_weight=0.36, expansion=1.70, core_efficiency=0.75,
        n_ms=3000, n_post=3000, t_max=15.0, x_end=1.0e-3,
        postms=True, homology=False,
        # --- white-dwarf parameters ---
        wd_mass=0.6, mu_e=2.0, A_ion=14.0, X_env=0.70, Z_env=0.0,
        Tc0=3.0e7, Tc_end=3.0e6, n_cool=4000,
        # --- neutron-star parameters ---
        eos="polytrope", n_mr=40, rho_lo=1.0e17, rho_hi=5.0e19,
        newtonian=False, p_nuc=None, gamma=None, m_observed=2.01,
        step_frac=0.01,
        # --- output ---
        outdir=None, csvdir=None, dpi=150, lw=1.6, no_plot=False):
    """
    Run one StellarEvolutionTracks calculation.

    mode selects the calculation:
        "tracks"  one evolutionary track
        "hr"      an HR diagram built from several tracks
        "wdcool"  white-dwarf structure and cooling
        "nsmr"    neutron-star mass-radius relation
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}.")

    dpi, lw = _validate_output(outdir, csvdir, dpi, lw)
    step_frac = _finite("step_frac", step_frac)
    if not (1.0e-5 < step_frac <= 0.1):
        raise ValueError(
            f"step_frac must lie in (1e-5, 0.1]; got {step_frac:g}.  Smaller "
            "values are more accurate and slower."
        )
    if no_plot and outdir is not None:
        raise ValueError("no_plot and outdir cannot both be requested.")
    if no_plot and csvdir is None:
        raise ValueError(
            "no_plot was requested but no csvdir was given, so the run would "
            "produce no output at all."
        )

    kw = dict(
        mass=mass, masses=masses, isochrones=isochrones,
        X=X, Z=Z, qc=qc, burning=burning, opacity=opacity,
        core_weight=core_weight, expansion=expansion,
        core_efficiency=core_efficiency,
        n_ms=n_ms, n_post=n_post, t_max=t_max, x_end=x_end,
        postms=postms, homology=homology,
        wd_mass=wd_mass, mu_e=mu_e, A_ion=A_ion,
        X_env=X_env, Z_env=Z_env, Tc0=Tc0, Tc_end=Tc_end, n_cool=n_cool,
        eos=eos, n_mr=n_mr, rho_lo=rho_lo, rho_hi=rho_hi,
        newtonian=newtonian, p_nuc=p_nuc, gamma=gamma,
        m_observed=m_observed, step_frac=step_frac,
        no_plot=no_plot,
    )

    print(f"[driver_sev] mode = {mode}")
    runner = {"tracks": _run_tracks, "hr": _run_hr,
              "wdcool": _run_wdcool, "nsmr": _run_nsmr}[mode]
    return runner(kw, outdir, csvdir, dpi, lw)
