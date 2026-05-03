"""
ADMM Worker — communicates with gRPC coordinator instead of MPI.Allreduce.
Run N instances of this script with different --rank flags.
"""
import grpc
import numpy as np
import pandas as pd
import glob, os, argparse, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'coordinator'))
import admm_pb2
import admm_pb2_grpc

P           = 26
STEP_SIZE   = 1.0
LOCAL_ITERS = 5
COORDINATOR = "localhost:50051"

def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-np.clip(t, -500, 500)))

def load_data(data_dir, rank, num_workers):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    local_files = [f for i, f in enumerate(files) if i % num_workers == rank]
    if not local_files:
        return np.zeros((0, P)), np.zeros(0)
    parts_X, parts_y = [], []
    for path in local_files:
        df = pd.read_csv(path)
        X = np.column_stack([np.ones(len(df))] + [df[f"x{i}"].values for i in range(1, 26)])
        parts_X.append(X); parts_y.append(df["y"].values)
    return np.vstack(parts_X), np.concatenate(parts_y)

def local_x_update(X, y, z, u, rho, step_size, iters):
    if X.shape[0] == 0:
        return np.zeros_like(z)
    x = z - u
    n = X.shape[0]
    for _ in range(iters):
        s = sigmoid(X @ x)
        grad = X.T @ (s - y) / n + rho * (x - z + u)
        x -= step_size * grad
    return x

def run(rank, num_workers, data_dir, max_iters, rho):
    X, y = load_data(data_dir, rank, num_workers)
    x = np.zeros(P); z = np.zeros(P); u = np.zeros(P)

    channel = grpc.insecure_channel(COORDINATOR)
    stub = admm_pb2_grpc.CoordinatorStub(channel)

    print(f"[Worker {rank}] Connected. Data shape: {X.shape}", flush=True)

    for k in range(max_iters):
        x = local_x_update(X, y, z, u, rho, STEP_SIZE, LOCAL_ITERS)
        x_plus_u = (x + u).tolist()

        response = stub.AggregateGradients(
            admm_pb2.LocalUpdate(rank=rank, x_plus_u=x_plus_u)
        )

        z = np.array(response.z)
        u = u + x - z

        if response.converged:
            print(f"[Worker {rank}] Converged at iteration {response.iteration}.", flush=True)
            break

    if rank == 0:
        print("\nFinal coefficients (z):")
        for j, val in enumerate(z):
            name = "intercept" if j == 0 else f"x{j}"
            print(f"  {name:<12} {val:>18.12f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank",        type=int,   required=True)
    parser.add_argument("--num-workers", type=int,   default=4)
    parser.add_argument("--data-dir",    type=str,   default="../data")
    parser.add_argument("--max-iters",   type=int,   default=200)
    parser.add_argument("--rho",         type=float, default=1.0)
    args = parser.parse_args()
    run(args.rank, args.num_workers, args.data_dir, args.max_iters, args.rho)
