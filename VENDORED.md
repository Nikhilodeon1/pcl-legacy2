# Vendored code

`config.py`, `src/`, and `pod_monitor.py` in this repo are copies of the
`reboot` project's code (the confound-detection methods paper's model,
training, and loss implementation), taken at the point this project split off
into its own repository (2026-09-10). They are not maintained in lockstep.

If `reboot`'s model/training/loss code changes in a way that matters here,
re-copy the relevant files manually — there is no automated sync.
