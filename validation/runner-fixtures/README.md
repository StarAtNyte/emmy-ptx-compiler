# GPU runner initialization regressions

These two fixtures use the same compiler-generated copy kernel: read x[0], write
sum_1[0]. Both allocate four f32 input values (first is 7) and one output. The zero
case names the input buffer x in zero_outputs; its independent expected output
is 0. The control omits that name and expects 7. These are GPU execution checks,
not CPU PTX snapshots; a code-generation snapshot cannot catch host initialization.

The old local runner passes the control but returns 7 for the zero case. The fixed
runner and hosted runner return 0. The initialization probe also checks returned
input flags, a fresh input value and timed local execution. See
../2026-10-24-runner-initialization.json for exact scope and hashes.

```sh
python tools/validate_isolated.py validation/runner-fixtures/*/ \
  --report artifacts/runner-fixtures.json
```

A local NVIDIA GPU/driver is required. Both kernels use safe fixed index 0 and
finite exact float32 values. This is a retained harness regression, not a compiler
counterexample or a proof of general argument initialization semantics.
