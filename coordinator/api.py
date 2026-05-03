"""
FastAPI wrapper — exposes /metrics and /health for the dashboard.
Proxies GetMetrics RPC from the running coordinator.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import grpc, sys, os

sys.path.insert(0, os.path.dirname(__file__))
import admm_pb2, admm_pb2_grpc

app = FastAPI(title="ADMM Coordinator API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

COORDINATOR = "localhost:50051"

@app.get("/metrics")
def get_metrics():
    try:
        channel = grpc.insecure_channel(COORDINATOR)
        stub = admm_pb2_grpc.CoordinatorStub(channel)
        m = stub.GetMetrics(admm_pb2.Empty())
        return {
            "iteration": m.iteration,
            "primal_residual": m.primal_residual,
            "dual_residual": m.dual_residual,
            "converged": m.converged,
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}
