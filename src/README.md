# Source Code

This directory is reserved for packaged or stable project code.

At the moment, active reusable code lives in `scripts/` because the workflow is
notebook-driven and still changing quickly. Move code here when it becomes a
stable library interface, for example:

- A package around feature extraction and artifact loading.
- A reusable training/evaluation CLI.
- Deployment-time scoring helpers for frozen two-tower embeddings.
- Shared data schemas or typed config objects.

Keep exploratory notebook-specific code in `notebooks/` until the interface is
settled.
