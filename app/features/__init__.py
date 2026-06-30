"""Redis feature store for the Recommendation Service.

Intentionally empty so importing the package has no side effects (it does not
pull the Redis/Mongo clients). Import the submodules directly.

CODE ONLY (Batch 6): nothing in this package is activated — it is not called
from any API route, not started by any background job, and not wired into the
gateway. It is the foundation that candidate generation / ranking will use in
later batches.
"""
