"""
gRPC Coordinator for ADMM — replaces MPI.Allreduce global barrier.
Collects local x+u updates from N workers, averages them into z (consensus).
"""
import grpc
import numpy as np
import threading
from concurrent import futures

import admm_pb2
import admm_pb2_grpc

NUM_WORKERS = 4
RHO = 1.0
MAX_ITERS = 200
ABS_TOL = 1e-1
REL_TOL = 1e-3
P = 26

class ADMMCoordinator(admm_pb2_grpc.CoordinatorServicer):
    def __init__(self):
        self.lock = threading.Lock()
        self.updates = {}
        self.z = np.zeros(P)
        self.z_prev = np.zeros(P)
        self.u_norms = {}
        self.iteration = 0
        self.converged = False
        self.primal_residual = float('inf')
        self.dual_residual = float('inf')
        self.barrier = threading.Barrier(NUM_WORKERS)

    def AggregateGradients(self, request, context):
        rank = request.rank
        x_plus_u = np.array(request.x_plus_u)

        with self.lock:
            self.updates[rank] = x_plus_u

        self.barrier.wait()

        with self.lock:
            if len(self.updates) == NUM_WORKERS:
                stacked = np.stack(list(self.updates.values()))
                z_new = stacked.mean(axis=0)

                self.primal_residual = float(
                    np.sum([np.linalg.norm(v - z_new) for v in stacked])
                )
                self.dual_residual = float(
                    RHO * np.linalg.norm(z_new - self.z) * NUM_WORKERS
                )

                x_norm_max = float(np.max([np.linalg.norm(v) for v in stacked]))
                u_norm_max = float(max(self.u_norms.values(), default=0))

                eps_pri  = (P**0.5) * ABS_TOL + REL_TOL * max(x_norm_max, np.linalg.norm(z_new))
                eps_dual = (P**0.5) * ABS_TOL + REL_TOL * u_norm_max

                self.converged = (
                    self.primal_residual < eps_pri and
                    self.dual_residual < eps_dual
                )
                self.z_prev = self.z.copy()
                self.z = z_new
                self.iteration += 1
                self.updates.clear()

                print(
                    f"iter {self.iteration:3d}: "
                    f"r={self.primal_residual:.4e}  s={self.dual_residual:.4e}  "
                    f"converged={self.converged}",
                    flush=True
                )

        return admm_pb2.ConsensusVector(
            z=self.z.tolist(),
            iteration=self.iteration,
            converged=self.converged
        )

    def GetMetrics(self, request, context):
        return admm_pb2.TrainingMetrics(
            iteration=self.iteration,
            primal_residual=self.primal_residual,
            dual_residual=self.dual_residual,
            converged=self.converged
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=NUM_WORKERS + 2))
    admm_pb2_grpc.add_CoordinatorServicer_to_server(ADMMCoordinator(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print(f"Coordinator listening on :50051  ({NUM_WORKERS} workers expected)", flush=True)
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
