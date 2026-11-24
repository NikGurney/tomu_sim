# ToM-U Reference Simulation

Reference implementation of the Theory of Mind Utility (ToM-U) for
*The Theory of Mind Utility: A Formal Account of Mentalizing*
([arXiv:2606.12721](https://arxiv.org/abs/2606.12721)).

The simulation is the consistency witness described in Appendix A of the
manuscript. It shows that the five definitions (§3.1–§3.5) compose into a
runnable system and that the six boundary conditions of Definition 5 can all
hold at once. The functional forms it uses (candidate generation, fit scoring,
confidence accumulation, residue dynamics, and so on) are illustrative choices.
They are not theoretical commitments and are not fitted to data (Appendix A.1).

## Quickstart

```
./run_all.sh          # macOS / Linux
run_all.bat           # Windows
```

Requires Python 3.9 or later and nothing else. The model, checks, tests, and all
TXT/CSV outputs use only the standard library. If Matplotlib is installed, Run 2
also writes the vector PDF for Figure A1; without it, the figure data is still
in the CSV.

`run_all` runs the two self-checks, the three demonstration runs, and the test
suite, and writes everything to `outputs/`. Each run asserts its own acceptance
criteria and stops with an error if any fail. The tests can also be run on their
own with `python -m checks.run_tests` (standard library) or `python -m pytest -q`.

## What reproduces what

| Script | Manuscript | Output |
|---|---|---|
| `checks/verify_residue.py` | Appendix A.3, Table A2 (Definition 5 boundary conditions) | console |
| `runs/run1_ablation.py` | Appendix A.4.1, Table A3 (source typing / discount ablation) | `outputs/run1_ablation.{txt,csv}` |
| `runs/run2_reconciliation.py` | Appendix A.4.2, Table A4, Figure A1 (min vs. weighted-average reconciliation) | `outputs/run2_reconciliation.{txt,csv,pdf}` |
| `runs/run3_faithfulness.py` | Appendix A.4.3–A.4.4, Tables A5–A6 (four variants; motivated episode) | `outputs/run3_faithfulness.{txt,csv}` |
| `checks/verify_structure.py` | Appendix A.4.5 (compositionality; depth-two projected node) | console |
| `tests/test_motivated_episode.py` | Appendix A.4.4 (residue-suppressed counterfactual; τ_commit = 0 limit) | test suite |

All parameters are in `tomu/config.py` and match Table A1. Analyses that change a
parameter (the depth-two check, the τ_commit = 0 limit) do so explicitly in the
script or test that runs them.

## Layout

```
tomu/        one module per definition
  lewm.py            Definition 1  local epistemic world model
  agent_node.py      Definition 2  agent node with ordered history
  proliferation.py   Definition 3  bounded agent proliferation
  inference.py       Definition 4  backward inference, self-projection,
                                   mutual reconciliation, motivated sessions
  residue.py         Definition 5  residue, credibility, rehabilitation
  scenarios.py       the popcorn-bag variants (§4) and the Run 2 setup
  config.py          all parameters (Table A1)
  trace.py           step-by-step inference transcripts
runs/        the three demonstration runs (Appendix A.4)
checks/      residue and structure self-checks, standard-library test runner
tests/       unit and integration tests
outputs/     generated results, committed for reference
```

## Determinism

There is no randomness in the model. Repeated runs give byte-identical TXT and
CSV files on any operating system, and the committed `outputs/` were produced by
`run_all.sh`. The PDF is pinned to a fixed creation date, but its bytes can
still vary between Matplotlib versions.

## Citation

If you use this code, please cite the manuscript:
[arXiv:2606.12721](https://arxiv.org/abs/2606.12721).
