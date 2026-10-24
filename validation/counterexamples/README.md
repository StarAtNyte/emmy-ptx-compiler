# Retained statement-reduction fixtures

The dated scope fixture retains a historical lexical-binding numerical failure:
11 → 10 statements after deleting unused arithmetic. The strided fixture remains
8 → 8 statements under the original output oracle. Both reach an observed
single-statement deletion fixed point; neither is claimed globally smallest.

Each directory contains typed IR, launch metadata/inputs and the original
independent expected arrays. Use `tools/local_runner.py` on either directory with
the current compiler for a passing check. Historical suspect compilers are
reproducible from `bb8d2c8:compile.py` (scope) and `488e04e:compile.py` (strided).
Numerical failures were rechecked in isolated contexts after reduction.

Full trial reports, compiler/reducer/input/oracle hashes and published file hashes
are in `../2026-10-24-statement-shrinking.json`. This evidence covers these two
fixtures and this deletion predicate, not general compiler safety or IR validity.
