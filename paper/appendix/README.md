# Appendix sources

LaTeX sources of the paper's appendix (state of 2026-10-08).

Build:

```bash
tectonic supplement.tex        # or: latexmk -pdf supplement.tex
```

| File | Section |
|---|---|
| `sections/01_construction.tex` | Construction and Implementation (inputs `sections/algorithm_training.tex`) |
| `sections/02_cohorts_evaluation.tex` | Cohorts and Evaluation |
| `sections/03_original_confirmation_larger_tasks.tex` | Original Confirmation and Larger Tasks |
| `sections/04_fresh_replication_action_content.tex` | Fresh Replication and Action Content |
| `sections/05_generated_controller_comparison.tex` | Generated-Controller Comparison |
| `sections/06_progress_conditions_advice_quantity.tex` | Progress Conditions and Advice Quantity |
| `sections/07_online_advice.tex` | Online Advice at a Fixed Request Budget |
| `sections/08_crafter.tex` | Additional Crafter Results |
| `sections/09_bank_robustness.tex` | Bank Regeneration and Aggregate Advice Rates |
| `sections/09b_bank_regeneration.tex` | Sensitivity to Bank Generation (sampled-action AUC) |
| `sections/10_earlier_methods.tex` | Earlier Action-Advising and Execution Comparisons |
| `sections/10b_online_advisors.tex` | Subsection: online advisors on all three tasks |
| `sections/11_baselines_timing.tex` | Fixed-Policy Baselines and Recorded Timing |

`09_bank_robustness.tex` and `09b_bank_regeneration.tex` both report the
DoorKey bank-regeneration study (greedy and sampled-action AUC); the final
appendix keeps one table for it. `10b_online_advisors.tex` belongs in
Section 10 after "Selective online action advising"; here it is appended at
the end of that section.

`additions/` holds two paragraphs that refer to the main-text table
`tab:bank_online_advisors` (bank versus online advisors through 5M) and are
therefore not compiled here.

`docs/EXPERIMENTS.md` at the repository root lists the code behind each
section.
