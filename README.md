# Distributed ADMM Logistic Regression: gRPC Coordinator Architecture

A distributed optimization system for large-scale logistic regression using the ADMM (Alternating Direction Method of Multipliers) algorithm. Workers run local gradient descent in parallel and synchronize through a gRPC coordinator service — no centralized data, no single point of computation.

Deployed originally on a 10-rank MPI cluster (SeaWulf HPC, 96-core partition). Architecture refactored to gRPC for heterogeneous deployment.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Distributed Communication | gRPC + Protocol Buffers |
| Optimization Algorithm | ADMM |
| Worker Runtime | Python, NumPy, pandas |
| Coordinator API | FastAPI, Uvicorn |
| HPC Deployment | MPI (mpi4py), SLURM |

## Repo Structure

    proto/admm.proto              - gRPC service definition
    coordinator/server.py         - gRPC coordinator
    coordinator/api.py            - FastAPI metrics layer
    worker/worker.py              - ADMM worker + gRPC client
    logistic_admm_mpi.py          - Original MPI implementation
    logistic_admm.slurm           - SLURM job script
    results/final_coefficients.txt - Converged model output

## Quickstart

Install dependencies:
    pip install grpcio grpcio-tools fastapi uvicorn numpy pandas

Compile the proto:
    python -m grpc_tools.protoc -I proto --python_out=coordinator --python_out=worker --grpc_python_out=coordinator --grpc_python_out=worker proto/admm.proto

Start coordinator:
    cd coordinator && python server.py

Start metrics API:
    cd coordinator && uvicorn api:app --port 8000

Launch 4 workers (one per terminal):
    python worker/worker.py --rank 0 --num-workers 4 --data-dir ./data
    python worker/worker.py --rank 1 --num-workers 4 --data-dir ./data
    python worker/worker.py --rank 2 --num-workers 4 --data-dir ./data
    python worker/worker.py --rank 3 --num-workers 4 --data-dir ./data

Monitor convergence:
    curl http://localhost:8000/metrics

HPC deployment:
    sbatch logistic_admm.slurm

## Results

| Config | Workers | Convergence |
|--------|---------|-------------|
| SeaWulf HPC (SLURM) | 10 MPI ranks | ~150 iterations |
| Local gRPC mode | 4 workers | ~4 iterations |

Top converged coefficients: x25=0.2429, x1=0.1727, x4=0.1164, intercept=0.1777
