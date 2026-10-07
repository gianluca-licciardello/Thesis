# Thesis

[MDERNet setup, exact experiment settings, input bundle, and run commands](MDERNet/README.md)
for AIDE Full/Clean/Balanced and PPB-Emo Full/Clean/EEG/EPQ/Cluster,
including independent and mixed PPB evaluation.

Run all 13 combinations with `bash current/run_all.sh` from `MDERNet`
after following the environment and input-bundle instructions.

The `MDERNet/current/` package implements the current 50-epoch,
validation-selected protocol, with LR 0.01 for AIDE and 0.0001 for PPB-Emo. Model weights and dataset inputs are shared
separately; the README includes an exporter and the full required-file list.
