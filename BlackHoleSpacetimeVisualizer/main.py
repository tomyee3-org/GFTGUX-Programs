"""
main.py
=======
Command-line entry point for Black_hole_spacetime_visualizer.

Four calculations share one program, chosen with --mode:

    embed      Flamm's paraboloid: the embedding diagram of the Schwarzschild
               spatial geometry
    tidal      radial (stretching) and tangential (compressing) tidal
               acceleration vs. radius, and an optional cross-mass
               comparison of an illustrative tidal-acceleration threshold
               against the horizon
    infall     a test particle dropped from rest and followed radially
               inward: proper time, distant-observer coordinate time, local
               speed, and the redshift of light it sends outward
    horizons   the event horizon vs. the apparent horizon of a black hole
               that gains mass by swallowing an infalling shell (a Vaidya
               spacetime), located by a numerical shooting method

All four calculations are exact consequences of the Schwarzschild geometry
(or, for horizons, the ingoing Vaidya generalisation of it) -- there is no
equation of state and no fitted closure here, unlike the stellar-structure
programs in this suite.  What is idealised is the physical scenario: point
test particles, purely radial motion, and, for horizons, a mass function
chosen by the student.

Examples
--------
  # Flamm's paraboloid for a 10 solar-mass black hole
  python main.py --mode embed --M 10

  # Tidal acceleration, comparing a stellar-mass hole with Sgr A*
  python main.py --mode tidal --M 10 --compare_masses 10,4.31e6

  # Radial infall released at 6 Schwarzschild radii, displayed on screen
  # AND saved as a PNG in ./runs, with a CSV written to ./data
  python main.py --mode infall --M 10 --r0_rs 6 --outdir ./runs --csvdir ./data

  # A black hole doubling in mass: event horizon vs. apparent horizon
  python main.py --mode horizons --M0 5 --M1 10
"""

import argparse

import driver_bh
import physics_bh


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
        prog="Black_hole_spacetime_visualizer",
        description=(
            "Visualise four aspects of Schwarzschild (and, for accreting "
            "holes, Vaidya) black-hole spacetime: the embedding diagram, "
            "tidal forces, time dilation and redshift for an infalling "
            "observer, and the distinction between the apparent and event "
            "horizons.  This is a transparent teaching model built from "
            "exact general-relativistic formulae, not a numerical-relativity "
            "code; every run prints exactly which geometry and which "
            "idealisation it used."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"Black_hole_spacetime_visualizer "
                f"{physics_bh.MODEL_VERSION} "
                f"(build {physics_bh.BUILD_ID})",
    )                
    p.add_argument("--mode", choices=driver_bh.MODES, default="embed",
                   help="which calculation to run")

    g = p.add_argument_group("Embedding diagram (mode: embed)")
    g.add_argument("--M", type=float, default=10.0, metavar="M_SUN",
                   help="black-hole mass [solar masses], shared by embed, "
                        "tidal and infall")
    g.add_argument("--r_max_rs", type=float, default=8.0, metavar="R/RS",
                   help="outer radius of the embedding diagram, in units of r_s")
    g.add_argument("--n_r", type=_positive_int, default=400, metavar="N",
                   help="radial points for embed or tidal")

    g = p.add_argument_group("Tidal acceleration (mode: tidal)")
    g.add_argument("--r_min_rs", type=float, default=1.01, metavar="R/RS",
                   help="innermost radius plotted, in units of r_s (must exceed 1)")
    g.add_argument("--separation", type=float, default=1.8, metavar="METRES",
                   help="proper length of the test rod / body, e.g. a human height")
    g.add_argument("--limit_g", type=float, default=100.0, metavar="G",
                   help="illustrative tidal-acceleration survival threshold, "
                        "in standard gravities g")
    g.add_argument("--compare_masses", type=str, default=None, metavar="LIST",
                   help="comma-separated masses [Msun] for the cross-mass "
                        "survival comparison, e.g. 10,4.31e6")

    g = p.add_argument_group("Radial infall (mode: infall)")
    g.add_argument("--r0_rs", type=float, default=6.0, metavar="R0/RS",
                   help="starting radius, released from rest, in units of r_s "
                        "(must exceed 1)")
    g.add_argument("--n_points", type=_positive_int, default=4000, metavar="N",
                   help="points recorded along the infall track")
    g.add_argument("--r_stop_rs", type=float, default=1.0005, metavar="R/RS",
                   help="radius at which the integration stops, in units of "
                        "r_s (must lie strictly between 1 and --r0_rs); the "
                        "Schwarzschild t coordinate used for the distant "
                        "observer's clock diverges at the horizon itself")
    g.add_argument("--step_frac", type=float, default=0.02, metavar="FRAC",
                   help="fractional step size for the infall integration, "
                        "in (1e-5, 0.2]")

    g = p.add_argument_group("Event vs. apparent horizon (mode: horizons)")
    g.add_argument("--M0", type=float, default=5.0, metavar="M_SUN",
                   help="initial black-hole mass")
    g.add_argument("--M1", type=float, default=10.0, metavar="M_SUN",
                   help="final black-hole mass, after accretion (must be >= M0)")
    g.add_argument("--v1_rs0", type=float, default=5.0, metavar="V/RS0",
                   help="advanced time at which accretion begins, in units "
                        "of the initial light-crossing time r_s0/c")
    g.add_argument("--duration_rs0", type=float, default=10.0, metavar="V/RS0",
                   help="duration of the accretion episode, in units of r_s0/c")
    g.add_argument("--v_start_margin_rs0", type=float, default=25.0, metavar="V/RS0",
                   help="how far before --v1_rs0 the shooting method starts "
                        "firing light rays; larger values reduce the residual "
                        "error in the located event horizon before v1")
    g.add_argument("--v_end_margin_rs0", type=float, default=15.0, metavar="V/RS0",
                   help="how far past the end of accretion the integration "
                        "and plot extend")
    g.add_argument("--n_steps", type=_positive_int, default=6000, metavar="N",
                   help="output-grid resolution in advanced time")
    g.add_argument("--bisect_iters", type=_positive_int, default=60, metavar="N",
                   help="bisection iterations used to locate the event horizon")
    g.add_argument("--n_family", type=_positive_int, default=9, metavar="N",
                   help="number of nearby geodesics drawn around the "
                        "event-horizon generator")

    g = p.add_argument_group("Output")
    g.add_argument("--outdir", type=str, default=None, metavar="PATH",
                   help="also save a timestamped PNG in PATH, IN ADDITION to "
                        "displaying the figure on screen (the screen display "
                        "is never suppressed by --outdir; use --no_plot to "
                        "skip the figure, and so the screen display, entirely)")
    g.add_argument("--csvdir", type=str, default=None, metavar="PATH",
                   help="also write timestamped CSV data files in PATH; has "
                        "no effect on whether the figure is shown on screen "
                        "or saved, and combines freely with --outdir and "
                        "--no_plot")
    g.add_argument("--no_plot", action="store_true",
                   help="skip the figure entirely -- no screen display and "
                        "no PNG, regardless of --outdir (requires --csvdir, "
                        "so the run still produces some output)")
    g.add_argument("--dpi", type=int, default=150, metavar="N",
                   help="PNG resolution")
    g.add_argument("--lw", type=float, default=1.6, metavar="PT",
                   help="curve line width [points]")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        driver_bh.run(
            mode=args.mode,
            M=args.M, r_max_rs=args.r_max_rs, n_r=args.n_r,
            r_min_rs=args.r_min_rs, separation=args.separation,
            limit_g=args.limit_g, compare_masses=args.compare_masses,
            r0_rs=args.r0_rs, n_points=args.n_points,
            r_stop_rs=args.r_stop_rs, step_frac=args.step_frac,
            M0=args.M0, M1=args.M1, v1_rs0=args.v1_rs0,
            duration_rs0=args.duration_rs0,
            v_start_margin_rs0=args.v_start_margin_rs0,
            v_end_margin_rs0=args.v_end_margin_rs0,
            n_steps=args.n_steps, bisect_iters=args.bisect_iters,
            n_family=args.n_family,
            outdir=args.outdir, csvdir=args.csvdir, no_plot=args.no_plot,
            dpi=args.dpi, lw=args.lw,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"Black_hole_spacetime_visualizer: {exc}") from exc


if __name__ == "__main__":
    main()
