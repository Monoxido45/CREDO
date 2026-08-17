# Paper Results

Filtered artifacts for the TMLR revision.

Configuration used for the final QNN results:

- fixed CREDO: `gamma = 0.2`
- adaptive CREDO: `gamma_min = 0.1`, `gamma_max = 0.9`, `tau_gamma = 1.0`
- CREDO dropout setting: `0.1`

Included tables:

- SMIS
- interval length
- marginal coverage
- outlier coverage, using LOF and Isolation Forest
- outlier/inlier interval-length ratio, using LOF and Isolation Forest
- fourth scarcity-score quartile coverage, using KNN and Isolation Forest

Included figures:

- outlier coverage/ratio heatmaps and summary barplots
- epistemic-uncertainty boxplots for inliers versus outliers, using LOF and Isolation Forest
- scarcity-Q4 coverage/SMIS heatmaps and summary barplots, using KNN and Isolation Forest
- fixed/adaptive gamma ablation curves

Raw experiment outputs and intermediate summaries remain in `../results/`.
