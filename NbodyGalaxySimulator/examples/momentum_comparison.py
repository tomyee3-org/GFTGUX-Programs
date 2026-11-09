"""Compare total-momentum changes in two runs starting from the same cluster."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import physics_nbg as phys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_bodies', type=int, default=40)
    parser.add_argument('--n_relax', type=float, default=0.5)
    parser.add_argument('--steps_per_crossing', type=int, default=30)
    parser.add_argument('--theta', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(f'N={args.n_bodies}, seed={args.seed}, n_relax={args.n_relax}, '
          f'steps_per_crossing={args.steps_per_crossing}, theta(tree)={args.theta}')
    print('Momenta are lab-frame totals in kg m/s; change is relative to the initial total.')
    print('The denominator is total mass times the initial mean particle speed (m/s).')
    print('method,initial_P_norm,final_P_norm,normalizer,final_change_fraction,max_sampled_change_fraction')
    for method in ('direct', 'tree'):
        result = phys.run_cluster(n_bodies=args.n_bodies, n_relax=args.n_relax,
                                  steps_per_crossing=args.steps_per_crossing,
                                  theta=args.theta, method=method, seed=args.seed)
        masses, velocities = result['masses'], result['velocities']
        momentum = np.sum(masses[None, :, None]*velocities, axis=1)
        normalizer = np.sum(masses)*np.mean(np.linalg.norm(velocities[0], axis=1))
        change = np.linalg.norm(momentum-momentum[0], axis=1)/normalizer
        print(f'{method},{np.linalg.norm(momentum[0]):.12g},'
              f'{np.linalg.norm(momentum[-1]):.12g},{normalizer:.12g},'
              f'{change[-1]:.12g},{np.max(change):.12g}')


if __name__ == '__main__':
    main()
