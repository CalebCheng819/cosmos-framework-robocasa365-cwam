# RoboCasa365 one-episode quick eval

This path exists only to obtain one raw exploratory episode quickly. It is
deliberately separate from the formal runner and evidence schema.

Every quick result is labeled:

- `formal_eligible=false`
- `asset_full_tree_verification_skipped=true`
- `lifecycle=raw_quick_nonsealed`

The authenticated publisher manifest and receipt are still SHA-256 checked,
but the live 24.1 GB asset payload is not fully enumerated or content-hashed.
The result therefore cannot be used as formal evidence, a prior-gate artifact,
or a defensible model-comparison success rate.

## H-cluster entrypoint

Create a fresh immutable source snapshot for the quick-eval commit first. The
snapshot must be a mode-0555 child of:

```text
/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/run_snapshots
```

Export these caller-provided values on the H login/workspace node:

```bash
export COSMOS_SOURCE=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/run_snapshots/<fresh-quick-snapshot>
export COSMOS_ENV=<pinned-cosmos-environment>
export ROBOCASA_ENV=<pinned-robocasa-environment>
export DEMIAN_REPO=<pinned-clean-demian-checkout>
export ROBOCASA_REPO=<published-clean-robocasa-checkout>
export ROBOSUITE_REPO=<pinned-clean-robosuite-checkout>
export SERVING_CHECKPOINT_PATH=<consolidated-serving-checkpoint>
export CHECKPOINT_MANIFEST=<serving-checkpoint-manifest>
export TRAINING_CONFIG=<exact-training-config>
export ACTION_STATS=<exact-action-stats>
export COSMOS_ENVIRONMENT_MANIFEST=<cosmos-environment-manifest>
export ROBOCASA_ENVIRONMENT_MANIFEST=<robocasa-environment-manifest>
export SOURCE_DCP_EXPORT_RECEIPT=<selected-serving-dcp-receipt>

export EVAL_EXPERIMENT_ID=QUICK-NONFORMAL-<unique-id>
export EVAL_LOGICAL_RUN_ID="${EVAL_EXPERIMENT_ID}-opendrawer-s42"
export EVAL_TASKS_CSV=OpenDrawer
export RJOB_NAME=<unique-lowercase-rjob-name>

export EXPECTED_SOURCE_COMMIT=<40-hex-quick-eval-commit>
export EXPECTED_SOURCE_MANIFEST_SHA256=<64-hex>
export EXPECTED_CHECKPOINT_MANIFEST_SHA256=<64-hex>
export EXPECTED_TRAINING_CONFIG_SHA256=<64-hex>
export EXPECTED_ACTION_STATS_SHA256=<64-hex>
export EXPECTED_COSMOS_ENVIRONMENT_MANIFEST_SHA256=<64-hex>
export EXPECTED_ROBOCASA_ENVIRONMENT_MANIFEST_SHA256=<64-hex>
export EXPECTED_SOURCE_DCP_EXPORT_RECEIPT_SHA256=<64-hex>
export EXPECTED_EGL_VENDOR_JSON_SHA256=<64-hex>
export EXPECTED_ROBOCASA_COMMIT=29f7ce8814c1547f5af762a0997fbd4b64848dd7
export EXPECTED_ROBOSUITE_COMMIT=<40-hex>
```

Then make exactly one state-changing submission call:

```bash
/usr/bin/bash "$COSMOS_SOURCE/examples/h_cluster/submit_cwam_cosmos3_robocasa365_quick_eval_rjob.sh"
```

The quick submitter has no predict phase and performs exactly one
`/usr/local/bin/rjob submit`. It requests one H200, 16 CPUs, 187500 MiB memory,
pod networking with host networking disabled, and node-local NVMe. It does not authenticate scheduler
acceptance or produce a formal runtime receipt.

On successful completion, inspect only:

```text
/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao/outputs/cwam-robocasa365-quick-eval/runs/<experiment-id>/<logical-run-id>/quick-nonformal-results/<run-name>/quick-result.json
```

Do not look for `summary.csv`, `task_success_rates.csv`, or
`evaluation_evidence.json`; this path intentionally never creates them.
