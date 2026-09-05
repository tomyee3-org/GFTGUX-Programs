"""
main.py
=======
Command-line entry point for NbodyGalaxySimulator.

Three calculations share one program, chosen with --mode:

    cluster  a Plummer sphere of stars, evolved for a chosen multiple of its
             own nominal two-body relaxation time -- seeing relaxation-
             driven expansion and (eventually) evaporation clearly requires
             tuned, non-default parameters, since the default softening
             actively suppresses both (see the Examples below and the Help
             file's EXP-1/EXP-11)
    galaxy   a cold, uniform sphere collapsing and "violently relaxing"
             (Lynden-Bell 1967), when it settles, into a quasi-equilibrium
             remnant -- a toy galaxy-formation experiment
    chaos    two indistinguishably close realizations of the same cluster,
             integrated identically, whose separation is tracked to
             produce a finite-time ESTIMATE of the e-folding (Lyapunov)
             growth rate of gravitational N-body motion, not a certified
             asymptotic value

All three share one engine: a Barnes-Hut octree (Barnes & Hut 1986) or,
for comparison, direct O(N^2) summation, with Plummer-softened gravity and
a kick-drift-kick (leapfrog) integrator. *Multiple*, the direct-summation
few-body program elsewhere in this project (which integrates with an
adaptive predictor-corrector scheme, not leapfrog), is assumed as a
prerequisite: this program does not re-teach Newtonian two-body motion
from scratch. It starts where Multiple's exact, unsoftened O(N^2)
approach stops being practical, trading Multiple's adaptive, per-step
accuracy for this program's fixed-timestep leapfrog scheme so that many
more bodies can be handled at once, and is deliberately a standalone
program rather than an added mode of Multiple, so as not to pull tree
codes and force softening into that program's introductory tutorial flow.

Examples
--------
  # A 200-star Plummer cluster at default parameters; whether two-body
  # relaxation is visibly acting within this run's length is not
  # guaranteed at default softening -- see the Help file's EXP-1 and
  # the tuned, multi-seed EXP-11 below for how to assess that properly
  python main.py --mode cluster

  # A smaller cluster with softening lowered so real unbound bodies can
  # appear (needs a smaller timestep to stay accurate -- see the Help
  # file); whether any given seed ends with a nonzero instantaneously-
  # unbound count varies -- see the Help file's EXP-11 for how to assess
  # this properly across several seeds
  python main.py --mode cluster --n_bodies 60 --softening_pc 0.0338 \
    --steps_per_crossing 150 --n_relax 40 --seed 0

  # A cold protogalactic sphere collapsing and violently relaxing
  python main.py --mode galaxy

  # Sensitivity to initial conditions: two 40-body realizations
  python main.py --mode chaos

  # Compare the tree force law against direct summation at fixed N
  python main.py --mode cluster --n_bodies 150 --method direct
"""

import argparse
import math

import driver_nbg
import physics_nbg


def _positive_int(text):
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer.") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be greater than zero.")
    return value


def _finite_float(text):
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number.") from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"{text!r} is not finite.")
    return value


def parse_args():
    p = argparse.ArgumentParser(
        prog="NbodyGalaxySimulator",
        description=(
            "Barnes-Hut tree-code N-body gravity with Plummer softening: "
            "star-cluster evaporation, cold-collapse galaxy formation, and "
            "a finite-time estimate of the Lyapunov (exponential-"
            "divergence) sensitivity of gravitational N-body motion. "
            "Assumes Multiple (direct-summation few-body gravity) as a "
            "prerequisite."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version",
                   version=(f"NbodyGalaxySimulator {physics_nbg.MODEL_VERSION} "
                            f"(build {physics_nbg.BUILD_ID})"))
    p.add_argument("--mode", choices=driver_nbg.MODES, default="cluster",
                   help="which calculation to run")

    g = p.add_argument_group("Bodies and mass (all modes)")
    g.add_argument("--n_bodies", type=_positive_int, default=None, metavar="N",
                   help="number of simulation particles "
                        f"(mode default: cluster=200, galaxy=300, chaos=40; "
                        f"hard limits {physics_nbg.MIN_BODIES}-"
                        f"{physics_nbg.MAX_BODIES:,})")
    g.add_argument("--total_mass_msun", type=_finite_float, default=None,
                   metavar="MSUN",
                   help="total mass [solar masses] (mode default: "
                        "cluster/chaos=1e3, galaxy=1e6)")

    g = p.add_argument_group("Cluster mode (Plummer sphere)")
    g.add_argument("--scale_radius_pc", type=_finite_float, default=None,
                   metavar="PC", help="Plummer scale radius [pc] "
                        "(cluster/chaos modes; default 1.0)")
    g.add_argument("--n_relax", type=_finite_float, default=None, metavar="X",
                   help="run length as a multiple of the initial two-body "
                        "relaxation time (default 5.0)")
    g.add_argument("--steps_per_crossing", type=_positive_int, default=None,
                   metavar="N", help="timesteps per initial crossing time "
                        "(cluster/chaos modes; default 60/50)")

    g = p.add_argument_group("Galaxy mode (uniform sphere, cold collapse)")
    g.add_argument("--radius_pc", type=_finite_float, default=None, metavar="PC",
                   help="initial sphere radius [pc] (default 200.0)")
    g.add_argument("--virial_ratio_init", type=_finite_float, default=None,
                   metavar="Q0", help="initial virial ratio 2T/|W_vir|; 0 = "
                        "perfectly cold, 1.0 = exact instantaneous scalar "
                        "virial balance (default 0.0)")
    g.add_argument("--n_freefall", type=_finite_float, default=None, metavar="X",
                   help="run length as a multiple of the initial free-fall "
                        "time (default 8.0)")
    g.add_argument("--steps_per_freefall", type=_positive_int, default=None,
                   metavar="N", help="timesteps per initial free-fall time "
                        "(default 80)")

    g = p.add_argument_group("Chaos mode (sensitivity to initial conditions)")
    g.add_argument("--relative_perturbation", type=_finite_float, default=None,
                   metavar="FRAC", help="initial relative position offset "
                        "between the two realizations (default 1e-8)")
    g.add_argument("--n_cross", type=_finite_float, default=None, metavar="X",
                   help="run length as a multiple of the initial crossing "
                        "time (default 120.0; chaos needs many more crossing "
                        "times than cluster/galaxy to show clean exponential "
                        "growth)")
    g.add_argument("--perturbation_seed", type=int, default=None, metavar="SEED",
                   help="seed for the random position offset (default: "
                        "unseeded)")

    g = p.add_argument_group("Gravity and integration (all modes)")
    g.add_argument("--softening_pc", type=_finite_float, default=None,
                   metavar="PC", help="Plummer softening length [pc] "
                        "(default: Athanassoula et al. 2000 optimal-"
                        "softening scaling, 0.98 a N^-0.26)")
    g.add_argument("--theta", type=_finite_float, default=None, metavar="THETA",
                   help="Barnes-Hut opening angle, "
                        f"{physics_nbg.MIN_THETA:g}-{physics_nbg.MAX_THETA:g} "
                        "(default 0.5); 0 forces an exact full-tree descent")
    g.add_argument("--method", choices=("tree", "direct"), default=None,
                   help="force evaluation method "
                        "(default: tree for cluster/galaxy, direct for chaos)")
    g.add_argument("--target_snapshots", type=_positive_int, default=None,
                   metavar="N", help="approximate number of recorded "
                        "snapshots (default: 150-200 depending on mode)")
    g.add_argument("--seed", type=int, default=None, metavar="SEED",
                   help="random seed for the initial-condition sampler "
                        "(default: unseeded, a different realization each run)")

    g = p.add_argument_group("Output")
    g.add_argument("--outdir", type=str, default=None, metavar="PATH",
                   help="also save a timestamped PNG (with a provenance "
                        "sidecar) in PATH; the figure is still displayed")
    g.add_argument("--csvdir", type=str, default=None, metavar="PATH",
                   help="also write a timestamped CSV diagnostics file in "
                        "PATH; this does not affect the screen display")
    g.add_argument("--no_plot", action="store_true",
                   help="skip the figure entirely (requires --csvdir, since "
                        "--outdir alone controls only the figure this skips)")
    g.add_argument("--dpi", type=int, default=150, metavar="N",
                   help="PNG resolution")
    g.add_argument("--lw", type=float, default=1.6, metavar="PT",
                   help="curve line width [points]")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        driver_nbg.run(
            mode=args.mode,
            n_bodies=args.n_bodies, total_mass_msun=args.total_mass_msun,
            scale_radius_pc=args.scale_radius_pc, radius_pc=args.radius_pc,
            virial_ratio_init=args.virial_ratio_init,
            n_relax=args.n_relax, n_freefall=args.n_freefall, n_cross=args.n_cross,
            steps_per_crossing=args.steps_per_crossing,
            steps_per_freefall=args.steps_per_freefall,
            relative_perturbation=args.relative_perturbation,
            softening_pc=args.softening_pc, theta=args.theta, method=args.method,
            target_snapshots=args.target_snapshots, seed=args.seed,
            perturbation_seed=args.perturbation_seed,
            outdir=args.outdir, csvdir=args.csvdir, no_plot=args.no_plot,
            dpi=args.dpi, lw=args.lw,
        )
    except (ValueError, RuntimeError, OSError, OverflowError) as exc:
        # physics_nbg.py converts the numerical-overflow conditions it
        # anticipates (see crossing_time() and run_galaxy()) into
        # ValueError with an explanatory message before they can
        # propagate this far. OverflowError is caught here too, as a
        # last-resort safety net, so that a parameter combination not
        # covered by one of those specific checks still exits cleanly
        # with a message instead of a raw traceback.
        raise SystemExit(f"NbodyGalaxySimulator: {exc}") from exc


if __name__ == "__main__":
    main()
