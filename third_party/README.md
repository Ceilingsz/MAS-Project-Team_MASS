# third_party

`STLPlanning/` is the authors' code for *Multi-agent Motion Planning from Signal Temporal Logic
Specifications* (Sun, Chen, Mitra, Fan; RA-L 2022): https://github.com/sundw2014/STLPlanning

It is **not vendored here**: the repository ships no LICENSE file, so we do not redistribute it.
Fetch it with `scripts/setup_stlplanning.sh` (pins commit `23f9551f63e86140fec622f14acc870ea7bd81bc`), or point the environment variable
`STL_PLANNING_PATH` at any checkout. It also needs Gurobi (`pip install gurobipy`; the pip licence is size-limited,
a free academic licence lifts the limit).
