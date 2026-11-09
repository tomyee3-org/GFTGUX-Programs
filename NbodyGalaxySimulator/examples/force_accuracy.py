"""Compare tree accelerations with direct summation on one initial configuration."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import physics_nbg as phys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_bodies', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not 3 <= args.n_bodies <= phys.MAX_BODIES:
        parser.error(f'--n_bodies must be in [3, {phys.MAX_BODIES}]')
    eps = phys.athanassoula_softening(args.n_bodies, phys.PC)
    initial = phys.plummer_sphere(args.n_bodies, 1000.0, 1.0,
                                  softening=eps, seed=args.seed)
    pos, masses = initial['positions'], initial['masses']
    direct = phys.compute_accelerations_direct(pos, masses, softening=eps)
    denominator = np.linalg.norm(direct, axis=1)
    if np.any(denominator == 0):
        raise ValueError('A direct acceleration is zero; its relative error is undefined.')
    print(f'N={args.n_bodies}, seed={args.seed}, a=1 pc, M=1000 Msun, '
          f'epsilon={eps/phys.PC:.10g} pc')
    print('theta,median_fraction,max_fraction,median_percent,max_percent')
    for theta in (0.0, 0.3, 0.5, 0.8, 1.0):
        tree = phys.compute_accelerations_tree(pos, masses, theta=theta, softening=eps)
        relative = np.linalg.norm(tree-direct, axis=1)/denominator
        median, maximum = np.median(relative), np.max(relative)
        print(f'{theta:.1f},{median:.12g},{maximum:.12g},'
              f'{100*median:.12g},{100*maximum:.12g}')


if __name__ == '__main__':
    main()
