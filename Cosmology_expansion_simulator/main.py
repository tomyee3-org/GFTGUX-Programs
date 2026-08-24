"""
main.py
=======
Command-line entry point for Cosmology_expansion_simulator.

Three calculations share one program, chosen with --mode:

    evolve    integrate a(t), H(t), the density parameters Omega_i(a),
              and the deceleration parameter q(t) for one FLRW cosmology
    compare   overlay several named or custom cosmologies and tabulate
              their ages
    age       scan the age of the universe (and related epochs) across
              one cosmological parameter, holding the others fixed

All three modes solve the same underlying Friedmann equation; only the
bookkeeping around it differs. The governing equations are exact within
the assumed FLRW perfect-fluid model; this program's implementation of
them is numerical and therefore has finite integration, root-finding,
and quadrature error. For an ordinary run, away from a_max/t_max landing
exactly on (or extremely near) a genuine turning point, that error is
typically ~1e-6 relative or better at the default step_frac, and tightens
with a smaller step_frac; see the help file for the benchmark comparisons
this is based on and for what changes near an event boundary. The
physical model itself is idealized, and the CPL dark-energy equation of
state w(a) = w0 + wa(1-a) is a phenomenological parametrization, not a
derived theory -- a placeholder for physics nobody yet understands.

Examples
--------
  # The concordance cosmology (Planck-2018-like flat LCDM). Omitting
  # --omega_de forces flatness exactly (Omega_DE0 = 1 - Omega_m0 - Omega_r0);
  # passing --omega_de 0.685 here instead would leave a stray, unintended
  # Omega_k0 = -9.24e-5 from the default radiation density.
  python main.py --mode evolve --H0 67.4 --omega_m 0.315

  # Einstein-de Sitter: matter only, flat -- the historical "age crisis" case
  python main.py --mode evolve --omega_m 1.0 --omega_de 0.0

  # A closed universe that recollapses; mirrored back down to a_i by default
  python main.py --mode evolve --omega_m 1.5 --omega_de 0.0 --continue_collapse

  # Phantom dark energy: w0 < -1 can lead to a future Big Rip, if no
  # earlier recollapse intervenes first
  python main.py --mode evolve --w0 -1.2 --a_max 60

  # Compare several textbook cosmologies side by side
  python main.py --mode compare --presets EdS,lambdaCDM,open,closed,phantom

  # The historical age problem: age of the universe vs Omega_m (flat)
  python main.py --mode age --scan_param omega_m --scan_lo 0.05 --scan_hi 1.5

  # Save a PNG and a CSV in addition to the on-screen display
  python main.py --mode evolve --outdir ./runs --csvdir ./data
"""

import argparse

import driver_cosmo
import physics_cosmo


def _positive_int(text):
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer.") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be greater than zero.")
    return value


def parse_args():
    p = argparse.ArgumentParser(
        prog="Cosmology_expansion_simulator",
        description=(
            "Integrate the Friedmann equation for a homogeneous, isotropic "
            "FLRW universe: scale-factor evolution a(t), the matter/"
            "radiation/curvature/dark-energy density budget, the Hubble "
            "parameter H(t), and the age of the universe for different "
            "cosmologies. The program evaluates the FLRW background "
            "equations for the specified components and solves the "
            "resulting expansion history numerically; the CPL dark-energy "
            "equation of state is a phenomenological parametrization, not "
            "a derived theory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version",
                   version=(f"Cosmology_expansion_simulator "
                            f"{physics_cosmo.MODEL_VERSION} "
                            f"(build {physics_cosmo.BUILD_ID})"))
    p.add_argument("--mode", choices=driver_cosmo.MODES, default="evolve",
                   help="which calculation to run")

    g = p.add_argument_group("Cosmological parameters (all modes)")
    g.add_argument("--H0", type=float, default=70.0, metavar="KM_S_MPC",
                   help="Hubble constant today [km/s/Mpc]")
    g.add_argument("--omega_m", type=float, default=0.30, metavar="FRAC",
                   help="matter density parameter today, Omega_m0")
    g.add_argument("--omega_r", type=float, default=9.24e-5, metavar="FRAC",
                   help="radiation density parameter today, Omega_r0 "
                        "(9.24e-5 is a representative photon+neutrino teaching "
                        "value; its precise size depends on H0, T_CMB, N_eff, "
                        "and the neutrino treatment)")
    g.add_argument("--omega_de", type=float, default=None, metavar="FRAC",
                   help="dark-energy density parameter today, Omega_DE0 "
                        "(default: None, which forces a FLAT universe by "
                        "setting Omega_DE0 = 1 - Omega_m0 - Omega_r0; give "
                        "an explicit value to get an open or closed model)")
    g.add_argument("--w0", type=float, default=-1.0, metavar="W0",
                   help="dark-energy equation of state AT a=1 (today); -1 "
                        "is a cosmological constant, <-1 contributes "
                        "toward phantom behavior, -1<w0<-1/3 is "
                        "quintessence-like, w0>=-1/3 does not accelerate "
                        "the expansion on its own -- if --wa is nonzero "
                        "this describes only today, not the whole history")
    g.add_argument("--wa", type=float, default=0.0, metavar="WA",
                   help="CPL evolution of the dark-energy equation of "
                        "state, w(a)=w0+wa(1-a); 0 means constant w0")

    g = p.add_argument_group("Integration control (modes: evolve, compare)")
    g.add_argument("--a_i", type=float, default=1.0e-8, metavar="A",
                   help="initial scale factor at which the numerical "
                        "integration starts (deep in radiation or matter "
                        "domination)")
    g.add_argument("--a_max", type=float, default=5.0, metavar="A",
                   help="stop the forward integration at this scale "
                        "factor, unless a recollapse or --t_max intervenes "
                        "first")
    g.add_argument("--t_max", type=float, default=None, metavar="GYR",
                   help="also stop the integration at this cosmic time "
                        "[Gyr] if reached before --a_max (default: no "
                        "time limit)")
    g.add_argument("--step_frac", type=float, default=0.005, metavar="FRAC",
                   help="Hubble-time-scaled RK4 step size, as a fraction "
                        "of the local Hubble time (roughly the fractional "
                        "change in ln(a) per step); this is a physically "
                        "motivated step size, not a true error-controlled "
                        "adaptive method, so check convergence by re-"
                        "running at a smaller value. Very small values "
                        "combined with a wide a_i-to-a_max range can "
                        "exceed the internal step-count safety limit; the "
                        "program will say so clearly if that happens")
    g.add_argument("--continue_collapse", action="store_true",
                   help="if the model recollapses, use the exact time-"
                        "reversal symmetry of the Friedmann equation to "
                        "also report the contracting branch, mirrored back "
                        "down to a_i (not all the way to the a=0 "
                        "singularity; the omitted interval near a=0 is "
                        "represented only by the analytic early-time "
                        "offset baked into the quoted total lifetime)")

    g = p.add_argument_group("Mode: compare")
    g.add_argument("--presets", type=str, default="EdS,lambdaCDM,closed,phantom",
                   metavar="LIST",
                   help="comma-separated preset cosmologies to overlay; "
                        f"choices are {', '.join(sorted(physics_cosmo.PRESETS))}, "
                        "or 'custom' to also include the cosmology "
                        "specified by the parameters above")

    g = p.add_argument_group("Mode: age (parameter scan)")
    g.add_argument("--scan_param", choices=driver_cosmo.SCAN_PARAMS,
                   default="omega_m",
                   help="which parameter to scan; the others are held at "
                        "the values given above")
    g.add_argument("--scan_lo", type=float, default=0.05, metavar="X",
                   help="lowest value of the scanned parameter")
    g.add_argument("--scan_hi", type=float, default=1.5, metavar="X",
                   help="highest value of the scanned parameter")
    g.add_argument("--scan_n", type=_positive_int, default=40, metavar="N",
                   help="number of points in the scan (3 to 500)")
    g.add_argument("--no_force_flat", dest="force_flat", action="store_false",
                   help="when scanning omega_m or omega_de, do NOT "
                        "compensate the other density to keep the model "
                        "flat -- let curvature vary instead (default: "
                        "flatness is preserved)")
    g.add_argument("--age_ref_gyr", type=float, default=13.0, metavar="GYR",
                   help="reference age [Gyr] drawn on the age-scan plot, "
                        "e.g. the age of the oldest globular clusters")

    g = p.add_argument_group("Output")
    g.add_argument("--outdir", type=str, default=None, metavar="PATH",
                   help="also save a timestamped PNG in PATH; the figure "
                        "is still displayed on screen")
    g.add_argument("--csvdir", type=str, default=None, metavar="PATH",
                   help="also write timestamped CSV data files in PATH; "
                        "this does not affect the screen display")
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
        driver_cosmo.run(
            mode=args.mode,
            H0=args.H0, omega_m=args.omega_m, omega_r=args.omega_r,
            omega_de=args.omega_de, w0=args.w0, wa=args.wa,
            a_i=args.a_i, a_max=args.a_max, t_max=args.t_max,
            step_frac=args.step_frac, continue_collapse=args.continue_collapse,
            presets=args.presets,
            scan_param=args.scan_param, scan_lo=args.scan_lo, scan_hi=args.scan_hi,
            scan_n=args.scan_n, force_flat=args.force_flat,
            age_ref_gyr=args.age_ref_gyr,
            outdir=args.outdir, csvdir=args.csvdir, no_plot=args.no_plot,
            dpi=args.dpi, lw=args.lw,
        )
    except (ValueError, RuntimeError, OSError, OverflowError) as exc:
        raise SystemExit(f"Cosmology_expansion_simulator: {exc}") from exc


if __name__ == "__main__":
    main()
