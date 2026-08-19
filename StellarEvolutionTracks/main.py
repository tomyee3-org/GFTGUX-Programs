"""
main.py
=======
Command-line entry point for StellarEvolutionTracks.

Four calculations share one program, chosen with --mode:

    tracks   evolution of a single star from the ZAMS to helium ignition
    hr       an HR diagram assembled from a grid of masses, with isochrones
    wdcool   white-dwarf structure and Mestel cooling
    nsmr     neutron-star mass-radius relation from the TOV equations

The four calculations are linked models of successive stages, not one
continuous integration: a track stops at helium ignition, and the remnant
it reports is a classification from the initial mass that you can then
hand to wdcool or nsmr yourself.

Examples
--------
  # Evolution of the Sun
  python main.py --mode tracks --mass 1.0

  # A 5 solar-mass star, saving both a figure and the data table
  python main.py --mode tracks --mass 5 --outdir ./runs --csvdir ./data

  # HR diagram with isochrones at 1, 5 and 10 Gyr
  python main.py --mode hr --isochrones 1,5,10

  # Cooling of a 0.6 solar-mass carbon-oxygen white dwarf
  python main.py --mode wdcool --wd_mass 0.6

  # Neutron-star mass-radius relation for an ideal degenerate neutron gas
  python main.py --mode nsmr --eos neutron

  # The same equation of state without general relativity
  python main.py --mode nsmr --eos neutron --newtonian
"""

import argparse

import driver_sev
import physics_sev


def _positive_int(text):
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer.") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be greater than zero.")
    return value


def _observed_mass(text):
    """--m_observed: a positive reference mass, or exactly 0 to omit it."""
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number.") from exc
    if value < 0.0:
        raise argparse.ArgumentTypeError(
            f"{value:g} is negative; use 0 to omit the reference line."
        )
    return value


def parse_args():
    p = argparse.ArgumentParser(
        prog="StellarEvolutionTracks",
        description=(
            "Integrate simple stellar-evolution equations, build HR-diagram "
            "tracks and isochrones, follow white-dwarf cooling curves, and "
            "solve the TOV equations for neutron-star mass-radius relations.  "
            "This is a transparent teaching model, not a stellar-evolution "
            "code: every run prints which of its ingredients were integrated "
            "and which were prescribed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version",
                   version=f"StellarEvolutionTracks {physics_sev.MODEL_VERSION}")
    p.add_argument("--mode", choices=driver_sev.MODES, default="tracks",
                   help="which calculation to run")

    g = p.add_argument_group("Star (modes: tracks, hr)")
    g.add_argument("--mass", type=float, default=1.0, metavar="M_SUN",
                   help="stellar mass for a single track [solar masses]; "
                        "accepted from 0.08 to 120, but the model is only "
                        f"pedagogically reliable from "
                        f"{physics_sev.TRUSTED_MASS_LO:g} to "
                        f"{physics_sev.TRUSTED_MASS_HI:g}")
    g.add_argument("--masses", type=str, metavar="LIST",
                   default="0.5,0.8,1.0,1.5,2.0,3.0,5.0,10.0",
                   help="comma-separated masses for the HR-diagram grid")
    g.add_argument("--isochrones", type=str, default=None, metavar="LIST",
                   help="comma-separated isochrone ages [Gyr], e.g. 1,5,10")
    g.add_argument("--X", type=float, default=0.70, metavar="FRAC",
                   help="hydrogen mass fraction")
    g.add_argument("--Z", type=float, default=0.02, metavar="FRAC",
                   help="metal mass fraction")

    g = p.add_argument_group("Stellar model closures (modes: tracks, hr)")
    g.add_argument("--qc", type=float, default=None, metavar="FRAC",
                   help="hydrogen-reservoir mass fraction (default: mass dependent)")
    g.add_argument("--burning", choices=("pp", "cno"), default=None,
                   help="nuclear energy-generation law (default: mass dependent)")
    g.add_argument("--opacity", choices=("thomson", "kramers"),
                   default="thomson", help="opacity law used in the homology relations")
    g.add_argument("--core_weight", type=float, default=0.36, metavar="W",
                   help="core weighting index w in mu_eff = mu_env^(1-w) mu_core^w")
    g.add_argument("--expansion", type=float, default=1.70, metavar="FACTOR",
                   help="main-sequence radius growth factor R_TAMS/R_ZAMS; "
                        "must be at least 1 (use 1 for no growth)")
    g.add_argument("--core_efficiency", type=float, default=0.75, metavar="FRAC",
                   help="helium-core mass at TAMS as a fraction of q_c M")
    g.add_argument("--homology", action="store_true",
                   help="use pure homology scalings for the ZAMS and the radius "
                        "response instead of the empirical fits")
    g.add_argument("--no_postms", dest="postms", action="store_false",
                   help="stop each track at the terminal-age main sequence")

    g = p.add_argument_group("Stellar integration control (modes: tracks, hr)")
    g.add_argument("--n_ms", type=_positive_int, default=3000, metavar="N",
                   help="main-sequence timesteps")
    g.add_argument("--n_post", type=_positive_int, default=3000, metavar="N",
                   help="post-main-sequence steps")
    g.add_argument("--t_max", type=float, default=15.0, metavar="GYR",
                   help="stop a track at this age if it is still burning "
                        "hydrogen; it does not limit the post-main-sequence "
                        "phase, and a track stopped this way reports no "
                        "main-sequence lifetime and no remnant timeline")
    g.add_argument("--x_end", type=float, default=1.0e-3, metavar="FRAC",
                   help="central hydrogen fraction that defines the TAMS")

    g = p.add_argument_group("White dwarf (mode: wdcool)")
    g.add_argument("--wd_mass", type=float, default=0.6, metavar="M_SUN",
                   help="white-dwarf mass [solar masses]")
    g.add_argument("--mu_e", type=float, default=2.0, metavar="MU",
                   help="mean molecular weight per electron of the degenerate "
                        "core, i.e. the mass per electron in atomic mass "
                        "units (about the number of nucleons per electron): "
                        "1 for hydrogen, 2 for helium, carbon or oxygen")
    g.add_argument("--A_ion", type=float, default=14.0, metavar="A",
                   help="mean ion mass number (14 for a carbon-oxygen mixture)")
    g.add_argument("--X_env", type=float, default=0.70, metavar="FRAC",
                   help="hydrogen fraction of the radiative envelope")
    g.add_argument("--Z_env", type=float, default=0.0, metavar="FRAC",
                   help="metal fraction of the envelope; controls the opacity")
    g.add_argument("--Tc0", type=float, default=3.0e7, metavar="K",
                   help="core temperature at the start of cooling")
    g.add_argument("--Tc_end", type=float, default=3.0e6, metavar="K",
                   help="core temperature at which cooling stops")
    g.add_argument("--n_cool", type=_positive_int, default=4000, metavar="N",
                   help="cooling integration steps")

    g = p.add_argument_group("Neutron star (mode: nsmr)")
    g.add_argument("--eos", choices=("polytrope", "neutron"),
                   default="polytrope",
                   help="equation of state: a toy stiff polytrope, or the "
                        "ideal degenerate neutron gas of Oppenheimer and "
                        "Volkoff")
    g.add_argument("--gamma", type=float, default=None, metavar="GAMMA",
                   help="polytropic index (default 2.5)")
    g.add_argument("--p_nuc", type=float, default=None, metavar="FRAC",
                   help="pressure at nuclear density as a fraction of rho c^2 "
                        "(default 0.04); larger means a stiffer star")
    g.add_argument("--newtonian", action="store_true",
                   help="use Newtonian hydrostatic equilibrium instead of TOV")
    g.add_argument("--n_mr", type=_positive_int, default=40, metavar="N",
                   help="number of stellar models along the sequence")
    g.add_argument("--rho_lo", type=float, default=1.0e17, metavar="KG_M3",
                   help="lowest central density")
    g.add_argument("--rho_hi", type=float, default=5.0e19, metavar="KG_M3",
                   help="highest central density")
    g.add_argument("--m_observed", type=_observed_mass, default=2.01,
                   metavar="M_SUN",
                   help="observational reference mass drawn on the first "
                        "panel, for comparison with measured high-mass "
                        "pulsars; 0 to omit it")

    g = p.add_argument_group("Structure integration control (modes: wdcool, nsmr)")
    g.add_argument("--step_frac", type=float, default=0.01, metavar="FRAC",
                   help="fractional step size for the structure integrations")

    g = p.add_argument_group("Output")
    g.add_argument("--outdir", type=str, default=None, metavar="PATH",
                   help="save a timestamped PNG in PATH instead of displaying it")
    g.add_argument("--csvdir", type=str, default=None, metavar="PATH",
                   help="also write timestamped CSV data files in PATH")
    g.add_argument("--no_plot", action="store_true",
                   help="skip the figure entirely (requires --csvdir)")
    g.add_argument("--dpi", type=int, default=150, metavar="N",
                   help="PNG resolution")
    g.add_argument("--lw", type=float, default=1.6, metavar="PT",
                   help="curve line width [points]")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        driver_sev.run(
            mode=args.mode,
            mass=args.mass, masses=args.masses, isochrones=args.isochrones,
            X=args.X, Z=args.Z, qc=args.qc,
            burning=args.burning, opacity=args.opacity,
            core_weight=args.core_weight, expansion=args.expansion,
            core_efficiency=args.core_efficiency,
            n_ms=args.n_ms, n_post=args.n_post,
            t_max=args.t_max, x_end=args.x_end,
            postms=args.postms, homology=args.homology,
            wd_mass=args.wd_mass, mu_e=args.mu_e, A_ion=args.A_ion,
            X_env=args.X_env, Z_env=args.Z_env,
            Tc0=args.Tc0, Tc_end=args.Tc_end, n_cool=args.n_cool,
            eos=args.eos, n_mr=args.n_mr,
            rho_lo=args.rho_lo, rho_hi=args.rho_hi,
            newtonian=args.newtonian, p_nuc=args.p_nuc, gamma=args.gamma,
            m_observed=(None if args.m_observed == 0 else args.m_observed),
            step_frac=args.step_frac,
            outdir=args.outdir, csvdir=args.csvdir, no_plot=args.no_plot,
            dpi=args.dpi, lw=args.lw,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"StellarEvolutionTracks: {exc}") from exc


if __name__ == "__main__":
    main()
