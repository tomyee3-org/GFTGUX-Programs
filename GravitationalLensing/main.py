"""
main.py
=======
Command-line entry point for GravitationalLensing.

Seven calculations share one program, chosen with --mode, and they follow
the teaching beats in the Help file:

    rays       side-view cartoon of why a ring splits into two images
    point      compact source through a point-mass lens
    blob       extended galaxy through a point-mass lens
    critical   Einstein ring, the point-caustic, and a side-view of the rays
    shear      SIS plus shear: diamond caustic and changing image count
    arcs       extended source on that diamond
    kappa      SIS convergence (the mass map, taught last)

Examples
--------
  python main.py --mode point
  python main.py --mode point --beta_x 0.0 --beta_y 0.0
  python main.py --mode shear --beta_x 0.55 --beta_y 0.10
  python main.py --mode critical --outdir ./runs
"""

import argparse
import os

import driver_gl
import physics_gl


def parse_args():
    p = argparse.ArgumentParser(
        prog="GravitationalLensing",
        description=(
            "Thin-lens teaching program: Einstein rings, doubles, diamond "
            "caustics, folds and cusps.  A transparent model, not a lens "
            "modelling code."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--version", action="version",
        version=(f"GravitationalLensing {physics_gl.MODEL_VERSION} "
                 f"(build {physics_gl.BUILD_ID})"),
    )
    p.add_argument("--mode", choices=driver_gl.MODES, default="point",
                   help="which drawing to make")

    g = p.add_argument_group("Source position [arcsec]")
    g.add_argument("--beta_x", type=float, default=None, metavar="ARCSEC",
                   help="unlensed source x (defaults depend on the mode)")
    g.add_argument("--beta_y", type=float, default=None, metavar="ARCSEC",
                   help="unlensed source y (defaults depend on the mode)")

    g = p.add_argument_group("Lens")
    g.add_argument("--logM", type=float, default=12.0, metavar="LOG10",
                   help="log10(M / M_sun) for point-mass modes")
    g.add_argument("--sigma_v", type=float, default=300.0, metavar="KM_S",
                   help="SIS velocity dispersion [km/s] for shear/arcs/kappa")
    g.add_argument("--gamma", type=float, default=0.25, metavar="SHEAR",
                   help="external shear for SIS+shear modes; |gamma| < 1")

    g = p.add_argument_group("Extended source")
    g.add_argument("--r_eff", type=float, default=0.25, metavar="ARCSEC",
                   help="effective radius of the elliptical Gaussian")
    g.add_argument("--q", type=float, default=0.60, metavar="RATIO",
                   help="axis ratio in (0, 1]")
    g.add_argument("--phi", type=float, default=30.0, metavar="DEG",
                   help="position angle [deg]")

    g = p.add_argument_group("Grid and output")
    g.add_argument("--n_pix", type=int, default=181, metavar="N",
                   help="pixels along one side of the image-plane grid")
    g.add_argument("--fov", type=float, default=6.0, metavar="ARCSEC",
                   help="field of view on a side")
    g.add_argument("--dpi", type=int, default=140, metavar="N",
                   help="PNG resolution")
    g.add_argument("--outdir", type=str, default=None, metavar="PATH",
                   help="also save a timestamped PNG (and provenance sidecar)")
    g.add_argument("--interactive", action="store_true",
                   help="open the figure on screen (no-op under MPLBACKEND=Agg)")
    return p.parse_args()


def _defaults_for(mode, args):
    """Fill source-position defaults that differ by teaching beat."""
    if args.beta_x is not None:
        bx = args.beta_x
    elif mode == "point":
        bx = 0.50
    elif mode in ("blob", "arcs"):
        bx = 0.18 if mode == "arcs" else 0.15
    elif mode == "shear":
        bx = 0.05
    else:
        bx = 0.00

    if args.beta_y is not None:
        by = args.beta_y
    elif mode == "blob":
        by = 0.05
    elif mode == "shear":
        by = 0.03
    elif mode == "rays":
        by = 0.35
    else:
        by = 0.00
    return bx, by


def _validate_args(args):
    physics_gl.validate_user_value("logM", args.logM)
    physics_gl.validate_user_value("sigma_v", args.sigma_v, positive=True)
    physics_gl.validate_user_value("gamma", args.gamma, exclusive_max=1.0)
    if args.gamma <= -1.0:
        raise ValueError(f"gamma must satisfy |gamma| < 1, got {args.gamma:g}")
    physics_gl.validate_user_value("r_eff", args.r_eff, positive=True)
    physics_gl.validate_user_value("q", args.q, min_value=1.0e-12, max_value=1.0)
    physics_gl.validate_user_value("phi", args.phi)
    physics_gl.validate_user_value("fov", args.fov, positive=True)
    physics_gl.validate_user_value("dpi", args.dpi, positive=True)
    if args.n_pix != int(args.n_pix) or int(args.n_pix) < 9:
        raise ValueError("n_pix must be an integer >= 9")
    if args.beta_x is not None:
        physics_gl.validate_user_value("beta_x", args.beta_x)
    if args.beta_y is not None:
        physics_gl.validate_user_value("beta_y", args.beta_y)


def main():
    args = parse_args()
    try:
        _validate_args(args)
    except ValueError as exc:
        raise SystemExit(f"GravitationalLensing: {exc}") from exc
    bx, by = _defaults_for(args.mode, args)
    show = args.interactive or args.outdir is None
    if os.environ.get("MPLBACKEND", "").lower() == "agg":
        show = False

    common = dict(outdir=args.outdir, dpi=args.dpi, show=show,
                  interactive=args.interactive)
    try:
        if args.mode == "rays":
            driver_gl.run_rays(beta_arcsec=by if args.beta_y is not None else 0.35,
                               **common)
        elif args.mode == "point":
            driver_gl.run_point(log10_m=args.logM, beta_x_arcsec=bx,
                                beta_y_arcsec=by, n_pix=args.n_pix,
                                fov_arcsec=args.fov, **common)
        elif args.mode == "blob":
            driver_gl.run_blob(log10_m=args.logM, beta_x_arcsec=bx,
                               beta_y_arcsec=by, r_eff_arcsec=args.r_eff,
                               q=args.q, phi_deg=args.phi,
                               n_pix=args.n_pix, fov_arcsec=args.fov,
                               **common)
        elif args.mode == "critical":
            driver_gl.run_critical(log10_m=args.logM, beta_x_arcsec=bx,
                                   beta_y_arcsec=by, n_pix=args.n_pix,
                                   fov_arcsec=args.fov, **common)
        elif args.mode == "shear":
            driver_gl.run_shear(sigma_v_kms=args.sigma_v, gamma=args.gamma,
                                beta_x_arcsec=bx, beta_y_arcsec=by,
                                **common)
        elif args.mode == "arcs":
            driver_gl.run_arcs(sigma_v_kms=args.sigma_v, gamma=args.gamma,
                               beta_x_arcsec=bx, beta_y_arcsec=by,
                               r_eff_arcsec=args.r_eff, q=args.q,
                               phi_deg=args.phi, n_pix=args.n_pix,
                               fov_arcsec=args.fov, **common)
        elif args.mode == "kappa":
            driver_gl.run_kappa(sigma_v_kms=args.sigma_v, n_pix=args.n_pix,
                                fov_arcsec=args.fov, **common)
    except (ValueError, RuntimeError, OSError, OverflowError,
            ZeroDivisionError) as exc:
        raise SystemExit(f"GravitationalLensing: {exc}") from exc


if __name__ == "__main__":
    main()
