# Distributed ADMM Logistic Regression — gRPC Coordinator Architecture

A distributed optimization system for large-scale logistic regression using ADMM (Alternating Direction Method of Multipliers). Workers run local gradient descent in parallel and synchronize through a gRPC coordinator service — no centralized data, no single point of computation.

Deployed originally on a 10-rank MPI cluster (SeaWulf HPC, 96-core partition). Architecture refactored to gRPC for heterogeneous deployment.

---

## Architecture

```
                        ADMM Training Job

  Worker 0  ---AggregateGradients(x+u)---->  +------------------+
  Worker 1  ---AggregateGradients(x+u)---->  |  gRPC            |
  Worker 2  ---AggregateGradients(x+u)---->  |  Coordinator     |
  Worker N  ---AggregateGradients(x+u)---->  |  :50051          |
            <------ConsensusVector(z)------  |                  |
                                             |  - Averages      |
                                             |    local updates  |
                                             |  - Tracks        |
                                             |    residuals     |
                                             |  - Broadcasts z  |
                                             +--------+---------+
                                                      |
                                               GetMetrics()
                                             +--------+---------+
                                             |  FastAPI         |
                                             |  Metrics API     |
                                             |  :8000           |
                                             |  GET /metrics    |
                                             |  GET /health     |
                                             +------------------+

  ADMM Consensus Step (per iteration):
    Each worker:  x_i = argmin L_i(x_i) + (rho/2)||x_i - z + u_i||^2
    Coordinator:  z   = (1/N) * sum(x_i + u_i)   <-- replaces MPI.Allreduce
    Each worker:  u_i = u_i + x_i - z
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Distributed Communication | gRPC + Protocol Buffers |
| Optimization Algorithm | ADMM |
| Worker Runtime | Python, NumPy, pandas |
| Coordinator API | FastAPI, Uvicorn |
| HPC Deployment | MPI (mpi4py), SLURM |

---

## Repo Structure

```
.
├── proto/
│   └── admm.proto                   gRPC service definition
├── coordinator/
│   ├── server.py                    gRPC coordinator — aggregates worker updates
│   ├── api.py                       FastAPI metrics layer (GET /metrics, /health)
│   ├── admm_pb2.py                  Generated protobuf message classes
│   └── admm_pb2_grpc.py             Generated gRPC stubs
├── worker/
│   └── worker.py                    ADMM worker — local gradient descent + gRPC client
├── logistic_admm_mpi.py             Original MPI implementation (SeaWulf HPC)
├── logistic_admm.slurm              SLURM job script (96-core partition)
├── results/
│   └── final_coefficients.txt       Converged model output (10 MPI ranks, ~150 iters)
└── requirements.txt
```

---

## Quickstart

### Prerequisites
```bash
pip install grpcio grpcio-tools fastapi uvicorn numpy pandas
```

### 1. Compile the proto
```bash
python -m grpc_tools.protoc \
  -I proto \
  --python_out=coordinator \
  --python_out=worker \
  --grpc_python_out=coordinator \
  --grpc_python_out=worker \
  proto/admm.proto
```

### 2. Start the coordinator
```bash
cd coordinator && python server.py
# Coordinator listening on :50051
```

### 3. Start the metrics API
```bash
cd coordinator && uvicorn api:app --port 8000
```

### 4. Launch workers (one terminal each)
```bash
python worker/worker.py --rank 0 --num-workers 4 --data-dir ./data
python worker/worker.py --rank 1 --num-workers 4 --data-dir ./data
python worker/worker.py --rank 2 --num-workers 4 --data-dir ./data
python worker/worker.py --rank 3 --num-workers 4 --data-dir ./data
```

### 5. Monitor live convergence
```bash
curl http://localhost:8000/metrics
# {"iteration": 3, "primal_residual": 0.071, "dual_residual": 0.0004, "converged": false}
```

### HPC Deployment (SLURM)
```bash
sbatch logistic_admm.slurm
```

---

## Results

| Config | Workers | Convergence |
|--------|---------|-------------|
| SeaWulf HPC (SLURM) | 10 MPI ranks | ~150 iterations |
| Local gRPC mode | 4 workers | ~4 iterations |

**Top converged coefficients (SeaWulf run):**

| Parameter | Estimate |
|-----------|----------|
| x25 | 0.2429 |
| x1 | 0.1727 |
| x4 | 0.1164 |
| intercept | 0.1777 |

---

## Why gRPC over MPI.Allreduce

MPI collectives require a homogeneous cluster — every rank must be alive and reachable. The gRPC coordinator decouples workers from the transport layer: workers can run on any machine, connect over a network, and the coordinator handles aggregation independently. Same ADMM math, production-grade communication.
