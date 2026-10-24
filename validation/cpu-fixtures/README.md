# CPU contract numerical fixtures

The signed-power-of-two boundary fixture was checked numerically on both local
and hosted GPUs before its PTX snapshot entered CPU CI. NumPy int64 reference
math observes quotient sign and low bits around signed INT_MIN/INT_MAX; inputs
and expected arrays remain unchanged from that independent oracle.

This is an optimization regression fixture, not a minimized historical failure.
GPU gate provenance and file hashes are in ../2026-10-24-production-iteration7.json.
CPU compilation/snapshot checks do not execute the kernel or replace those gates.
