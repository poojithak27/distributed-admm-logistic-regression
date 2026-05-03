#!/usr/bin/env python3
"""
Distributed Logistic Regression with ADMM using mpi4py.

Reads all .csv partitions from a shared data directory, distributes
them round-robin across MPI ranks, and solves for one consensus
coefficient vector beta (intercept + x1..x25) via ADMM.

Usage:
    mpirun -np 10 python3 logistic_admm_mpi.py
"""

from mpi4py import MPI
import numpy as np
import pandas as pd
import os
import glob

# ---------------------------------------------------------------------------
# MPI setup
# ---------------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = "/gpfs/projects/AMS598/projects2025_data/project3_data"
rho        = 1.0
max_iters  = 200
abs_tol    = 1e-3
rel_tol    = 1e-3
step_size  = 1.0
p          = 26


def sigmoid(t: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(t, -500, 500)))


def get_all_csv_files() -> list:
    """Return a sorted list of every .csv file in DATA_DIR."""
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    files.sort()
    return files


def load_local_data():
    """
    Round-robin file assignment: file i goes to rank (i mod size).
    """
    all_files   = get_all_csv_files()
    local_files = [f for i, f in enumerate(all_files) if i % size == rank]

    if not local_files:
        return np.zeros((0, p)), np.zeros(0)

    X_parts, y_parts = [], []
    for path in local_files:
        df = pd.read_csv(path)
        X  = np.column_stack(
            [np.ones(len(df))] + [df[f"x{i}"].values for i in range(1, 26)]
        )
        y  = df["y"].values
        X_parts.append(X)
        y_parts.append(y)

    return np.vstack(X_parts), np.concatenate(y_parts)


def local_x_update(X, y, z, u, rho, step_size, iters=5):
    """
    Approximately solve the local ADMM x-subproblem via gradient steps.
    """
    if X.shape[0] == 0:
        return np.zeros_like(z)

    x = z - u
    n = X.shape[0]
    for _ in range(iters):
        s          = sigmoid(X @ x)
        grad_logit = X.T @ (s - y) / n
        grad_quad  = rho * (x - z + u)
        x          = x - step_size * (grad_logit + grad_quad)
    return x


def admm_logistic_regression():
    X, y = load_local_data()

    x = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)

    for k in range(max_iters):
        x = local_x_update(X, y, z, u, rho, step_size, iters=5)

        x_plus_u = x + u
        summed   = np.zeros_like(x_plus_u)
        comm.Allreduce(x_plus_u, summed, op=MPI.SUM)
        z_new = summed / size

        u = u + x - z_new

        r_local = np.linalg.norm(x - z_new)
        s_local = np.linalg.norm(rho * (z_new - z))

        r_norm = comm.allreduce(r_local, op=MPI.SUM)
        s_norm = comm.allreduce(s_local, op=MPI.SUM)

        x_norm_max = comm.allreduce(np.linalg.norm(x), op=MPI.MAX)
        u_norm_max = comm.allreduce(np.linalg.norm(u), op=MPI.MAX)

        eps_pri  = np.sqrt(p) * abs_tol + rel_tol * max(x_norm_max, np.linalg.norm(z_new))
        eps_dual = np.sqrt(p) * abs_tol + rel_tol * u_norm_max

        z = z_new

        if rank == 0 and (k % 10 == 0 or k == max_iters - 1):
            print(f"iter {k:3d}: r={r_norm:.4e}, s={s_norm:.4e}", flush=True)

        if r_norm < eps_pri and s_norm < eps_dual:
            if rank == 0:
                print(f"\nConverged at iteration {k}.")
            break

    if rank == 0:
        print("\nFinal consensus coefficients (beta):")
        print(f"{'Parameter':<12} {'Estimate':>18}")
        print("-" * 32)
        for j, val in enumerate(z):
            name = "intercept" if j == 0 else f"x{j}"
            print(f"{name:<12} {val:>18.12f}")

    return z


if __name__ == "__main__":
    admm_logistic_regression()
