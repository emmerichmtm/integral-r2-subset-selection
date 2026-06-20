# Fast Exact Algorithms for Integral R2 Subset Selection (Version V17)

This bundle contains an arXiv-style manuscript and a pure-standard-library Python reference implementation for fixed-cardinality subset selection under the exact integral bi-objective `R2` indicator.

## Files

- `r2_lr_dp_subset_selection_v17.tex` - LaTeX manuscript source.
- `r2_lr_dp_subset_selection_v17.pdf` - compiled manuscript PDF.
- `r2_subset_dp_v17.py` - Python implementation.
- `dp_test_output_v17.txt` - reproducibility output from the Python script.
- `PRIMEarxiv.sty` - style file used by the manuscript.
- `README_V17.md` - this file.

## New in V17

Version V17 adds a dedicated section on the matrix-search route:

- preconditions for the staircase totally monotone layer matrix;
- a lower-envelope matrix-search algorithm;
- correctness proof via staircase total monotonicity and single crossing;
- complexity analysis giving `O(n)` time per DP layer and `O(k n)` total time;
- a gentle appendix introducing matrix search.

The Python bundle now includes `dp_select_exact_r2_matrix_search(points, k)` and checks it against the direct LRDP and divide-and-conquer DP.

## Algorithms in the Python file

- `brute_force_select_exact_r2` for small verification cases;
- `dp_select_exact_r2` for the direct `O(k n^2)` Bellman DP;
- `dp_select_exact_r2_divide_conquer` for the Monge divide-and-conquer DP, `O(k n log n)`;
- `dp_select_exact_r2_matrix_search` for the lower-envelope matrix-search DP, `O(k n)` under the arithmetic model used in the paper.

## Reproduce

```bash
python3 r2_subset_dp_v17.py > dp_test_output_v17.txt
pdflatex -interaction=nonstopmode r2_lr_dp_subset_selection_v17.tex
pdflatex -interaction=nonstopmode r2_lr_dp_subset_selection_v17.tex
```

## Notes

- The mathematical matrix-search proof assumes exact arithmetic comparisons and constant-time crossing computations.
- The Python implementation uses floating-point arithmetic with tolerances and local neighbor checks for robust tie behavior.
- Candidate points are minimization points measured relative to a shifted utopian point and represented as a sorted bi-objective Pareto-front approximation.


## V17 appendix update

Version V17 makes the manuscript more arXiv-ready: it shortens the title, updates the running head, adds code availability with a GitHub placeholder, adds acknowledgements and a declaration on generative AI use, updates the section overview, and keeps the expanded matrix-search appendix. It adds a brief introduction to Monge matrices and total monotonicity, cites both the original SMAWK/matrix-search paper and a Monge-properties survey, and includes a small numerical example plus a staircase-domain schematic.


## License

This work is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

You are free to share and adapt the material for any purpose, provided appropriate credit is given.

See the full license here: https://creativecommons.org/licenses/by/4.0/
