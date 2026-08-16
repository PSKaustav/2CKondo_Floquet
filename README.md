# 2CKondo_Floquet
This repository houses the codes corresponding to the numerical simulations carried out as part of a project pursued by Prof Elio König, Prof Thomas Ayral, Soumyadeep Sarma and myself, Kaustav Prakash Sarma, where we are studying the Floquet dynamics of the 2 channel Kondo Circuit, specifically, 'Emergent Floquet Anyons in a 2 channel Kondo Circuit'.

## SERGIO notes-driven comparison workflow

Use the notes-driven comparison CLI to generate:
1. benchmark vs no-truncation vs chosen bond-dimension plot,
2. chi sweep plot,
3. per-step CSV export.

From `Trials/`:

```bash
python sergio_notes_cli.py \
  --benchmark-file "N = 6, theta = 1.05, theta_k = 0.79, t = 100_sz_tol.txt" \
  --primary-chi 50 \
  --chi-sweep 10,20,50,100 \
  --three-plot compare_three_curves.png \
  --sweep-plot compare_chi_sweep.png \
  --series-csv sergio_comparison_series.csv
```

Notes:
- By default, this uses canonical benchmark angles (`theta = pi/3`, `theta_k = pi/4`) even though the benchmark file name is rounded (`1.05`, `0.79`).
- Use `--use-rounded-angles` if you explicitly want the rounded values.
- The no-truncation curve is computed with `do_orbital_truncation=False` and `bond_dim=inf`.

Regression check:

```bash
python tests/sergio_notes_regression.py
```
