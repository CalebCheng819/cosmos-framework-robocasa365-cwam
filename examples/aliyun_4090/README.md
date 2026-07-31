# Alibaba DSW RoboCasa365 formal evaluation

This directory contains the fail-closed adapter for the CWAM/Cosmos3
RoboCasa365 formal evaluation on `root@123.57.187.96:974`. It preserves the
research protocol while mapping the deployment to the verified single-node
Alibaba PAI DSW allocation:

- one physical server;
- four visible NVIDIA 4090 GPUs with compute capability 8.9 (`sm89`);
- 128 effective CPUs;
- a 400 GiB cgroup memory limit;
- one four-rank FSDP policy server, with rank 0 serving one loopback WebSocket
  endpoint; and
- eight logical evaluator shards, run sequentially, covering 17 atomic tasks
  with 20 episodes per task (exactly 340 primary-metric episodes).

The runner first performs one separate closed-loop `CloseBlenderLid` episode.
That preflight must finish with at least one finite width-7 policy action, but
its success bit is not used as an admission criterion and the episode is
explicitly excluded from the 340-episode primary metric. The formal eight-shard
loop starts only after this gate passes.

## Audited immutable inputs

The following hashes are fixed by the H export and the assets already verified
on DSW. Paths shown here are the live DSW canonical paths where known.

| Input | SHA-256 or commit |
| --- | --- |
| H serving checkpoint manifest | `7bcb9494b9f00b93dbad7d2b533032d3c8086e134986ae70f1c03abb1858f500` |
| serving model tree | `77d2a84a35e81438c87a6e31bedc09299b294fcbe9841f4e943b985bbcb6f846` |
| EMA export receipt | `26bc5f658aa71a2ac8018dabb53574c9917171bb69bff9e4091d5621961c4200` |
| EMA key manifest | `e2edc73c06b628bb9294b157af1f6d3be3ec11c626713a709c9f64667ca2d8fa` |
| EMA validation receipt | `887d7a962bda3f37f105d210945ec21bebed4becb0c42677b8c4ead256b673dd` |
| independent full source-DCP verification | `1f9352ff4d97116d4861e2fec6dc5522a738a5ac58d5e15bf0025f3b1f45f646` |
| EMA Ceph transfer manifest | `1f00b9d45f1d7ca2e6691fcd117d9da42c3444f9e517f5f3091362e1b3992f9b` |
| R14 training config | `c8269c4f12c99ac98e71e85334933d9749cd66a494960d6afe6e91368134caaa` |
| RoboCasa365 action stats | `2b48eb8a42e0c9c7befc45034d152282dad328f36389c10cafdc5be2635b67df` |
| official Wan2.2 VAE | `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36` |
| Qwen tokenizer external manifest | `c71787b714a6876114b69d249e88bfa50de1262d33f46f8285dd00da729d4978` |
| asset/environment closure manifest | `08b73f93a2ef191e2d82783e0a35689d1416a6d05e0384145e78551b3103f5bb` |
| H closure verification receipt | `60263e2de95efeeab89bc04b0b176331440a46ef8f027c8d708f8a9ecf739d45` |
| Cosmos environment-spec source | `09f23119ea92c707207bba55565e7a09d16896a2` |
| DeMiAn | `2600ece5bf0eeacf010a95f084dcda2de6d30d59` |
| RoboCasa | `29f7ce8814c1547f5af762a0997fbd4b64848dd7` |
| robosuite | `232ce7d4a6ed89c949a9aba024a05c8c32fdd08b` |

Verified live DSW files:

```text
/mnt/workspace/cwam_cosmos3_robocasa365_eval/inputs/config/R14-config.yaml
/mnt/workspace/cwam_cosmos3_robocasa365_eval/inputs/config/robocasa365_native_frame_wise_relative_rot6d.json
/mnt/workspace/cwam_cosmos3_robocasa365_eval/inputs/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
/mnt/workspace/cwam_cosmos3_robocasa365_eval/inputs/models/Qwen3-VL-8B-Instruct-tokenizer
/mnt/workspace/cwam_cosmos3_robocasa365_eval/inputs/config/Qwen3-VL-8B-Instruct-tokenizer-manifest.json
```

The checkpoint transfer is still staged at the time of writing. Its only
admissible final model path, after complete shard verification and atomic
publication, is:

```text
/mnt/workspace/cwam_cosmos3_robocasa365_eval/inputs/checkpoints/R14-iter_000010000-serving-ema-bf16-v1/model
```

Do not point the runner at the `.incomplete-*` transfer directory.

The Qwen manifest must remain outside the tokenizer directory. The tokenizer
directory must contain exactly the regular files listed by the manifest, with
no symlinks or unmanifested files.

The policy server imports
`openpi.serving.websocket_policy_server.WebsocketPolicyServer` directly from
the pinned DeMiAn `openpi/src` tree. Runtime evidence binds that exact source
path and SHA-256; an independently installed `openpi_server` wheel cannot
silently select a different WebSocket implementation.

The EMA serving checkpoint is not an ordinary model copy. Its receipts bind
selection of `net_ema.*` only: 809 keys, 6,472 chunks, FP32-to-BF16 conversion,
and eight DCP shards. The runner hashes the complete destination tree, verifies
it against the H tree hash, then creates a new DSW-path-bound checkpoint
manifest. The H manifest remains source provenance and is never treated as
portable path metadata.

## Native DSW environments

Formal evaluation uses native exact rebuilds on DSW, not a relocated H virtual
environment.

`$COSMOS_ENV/.cwam_env_ready.json` must be mode `0444` and contain exactly:

```text
schema_version, status, python, torch, torch_cuda, cuda_available, packages,
source_head, pyproject_sha256, uv_lock_sha256, uv, default_index, cudart_alias
```

It must bind `source_head=09f23119ea92c707207bba55565e7a09d16896a2`,
PyTorch `2.10.0+cu128`, CUDA `12.8`, uv `0.11.28`, and a live four-GPU CUDA
probe. This environment-spec commit is deliberately distinct from
`EXPECTED_SOURCE_COMMIT`, which identifies the final immutable evaluation code
snapshot.

`$ROBOCASA_ENV/.cwam_env_ready.json` must be mode `0444` and contain exactly:

```text
schema_version, status, python, torch, torch_cuda, cuda_available, mujoco,
robocasa, robosuite, distribution_count, distributions_sha256,
origin_environment_manifest_sha256, origin_requirements_sha256,
relocated_requirements_sha256, robocasa_commit, robosuite_commit,
package_source_note
```

It must bind Python `3.11.14`, PyTorch `2.7.1+cu128`, CUDA `12.8`, MuJoCo
`3.3.1`, RoboCasa `1.0.1`, robosuite `1.5.2`, and the commits above. The
verifier also compares the marker against the live interpreter, PyTorch/CUDA
runtime, imported simulator package versions, and the complete canonical
installed-distribution count and digest. The digest is SHA-256 over compact,
ASCII, key-sorted JSON mapping each normalized PEP 503 distribution name to
`{"name": <metadata Name>, "version": <installed version>}`.

The historical H Cosmos marker
`14968ffebfb1f885fd799257513c9cb1588c7c85a07d43e44ef78a123da9c99d`
is provenance only and must not be supplied as a DSW-native marker. The final
source snapshot SHA and both DSW-native marker SHA values are launch-time
values: generate them after the final source snapshot and native environment
rebuilds, then pass their actual hashes. Do not substitute an older R18 value.

## Launch contract

All transferred archives and the 17-object EMA transfer must be downloaded and
fully verified before launching. The runner does not accept a remote-size-only
claim as proof that a DSW destination file is intact. Receipt JSON and source
manifests used as formal inputs must be immutable mode `0444`.

Use a new experiment ID, logical run ID, source snapshot, and output directory.
`EVAL_OUTPUT_ROOT` must equal
`$DSW_CPFS_OUTPUT_PARENT/$EVAL_EXPERIMENT_ID/$EVAL_LOGICAL_RUN_ID`; the entire
experiment output root must not already exist. `DSW_CPFS_MOUNT` is checked with
`findmnt` and must identify the exact CPFS mount. Do not guess these three
storage paths: record the live mount target selected for the run.

A launch shell should define the following groups of variables. Values in angle
brackets are intentionally unresolved until the final snapshot, native
environment rebuild, or CPFS mount selection exists.

```bash
export INPUTS_ROOT=/mnt/workspace/cwam_cosmos3_robocasa365_eval/inputs

export COSMOS_SOURCE=<immutable-final-source-snapshot>
export SOURCE_MANIFEST="$COSMOS_SOURCE/SNAPSHOT_MANIFEST.json"
export EXPECTED_SOURCE_COMMIT=<final-40-hex-source-commit>
export EXPECTED_SOURCE_MANIFEST_SHA256=<final-snapshot-manifest-sha256>
export EXPECTED_COSMOS_ENV_SOURCE_COMMIT=09f23119ea92c707207bba55565e7a09d16896a2

export COSMOS_ENV="$INPUTS_ROOT/envs/cosmos-framework-09f23119-cu128-train"
export ROBOCASA_ENV="$INPUTS_ROOT/envs/robocasa-1.0.1-py311-mj331-rs232ce7-shared"
export COSMOS_ENV_READY_MARKER="$COSMOS_ENV/.cwam_env_ready.json"
export ROBOCASA_ENV_READY_MARKER="$ROBOCASA_ENV/.cwam_env_ready.json"
export EXPECTED_COSMOS_ENV_READY_MARKER_SHA256=<new-dsw-native-marker-sha256>
export EXPECTED_ROBOCASA_ENV_READY_MARKER_SHA256=<new-dsw-native-marker-sha256>
export ROBOCASA_TORCHCODEC_RELAY_RECEIPT="$INPUTS_ROOT/config/robocasa-torchcodec-plain-wheel-relay-receipt.json"
export EXPECTED_ROBOCASA_TORCHCODEC_RELAY_RECEIPT_SHA256=1b16cc5419b7eb8558d5c0bd6dc4099493df629d50756ad3116b3c1da47f5ce4

export DEMIAN_REPO="$INPUTS_ROOT/demian-2600ece5bf0e"
export ROBOCASA_REPO="$INPUTS_ROOT/repos/robocasa-29f7ce8814c1"
export ROBOCASA_ASSET_ROOT="$ROBOCASA_REPO"
export ROBOSUITE_REPO="$INPUTS_ROOT/robosuite-232ce7d4a6ed"
export EXPECTED_DEMIAN_COMMIT=2600ece5bf0eeacf010a95f084dcda2de6d30d59
export EXPECTED_ROBOCASA_COMMIT=29f7ce8814c1547f5af762a0997fbd4b64848dd7
export EXPECTED_ROBOSUITE_COMMIT=232ce7d4a6ed89c949a9aba024a05c8c32fdd08b

export CHECKPOINT_ARTIFACT_ROOT="$INPUTS_ROOT/checkpoints/R14-iter_000010000-serving-ema-bf16-v1"
export SERVING_CHECKPOINT_PATH="$CHECKPOINT_ARTIFACT_ROOT/model"
export H_SERVING_CHECKPOINT_MANIFEST="$CHECKPOINT_ARTIFACT_ROOT/h-serving-checkpoint-manifest.json"
export EMA_EXPORT_RECEIPT="$CHECKPOINT_ARTIFACT_ROOT/ema-bf16-export-receipt.json"
export EMA_KEY_MANIFEST="$CHECKPOINT_ARTIFACT_ROOT/ema-bf16-key-manifest.json"
export EMA_VALIDATION="$CHECKPOINT_ARTIFACT_ROOT/ema-bf16-validation.json"
export EMA_SOURCE_DCP_FULL_RECEIPT="$CHECKPOINT_ARTIFACT_ROOT/source-dcp-full-verification.json"
export EMA_TRANSFER_MANIFEST="$CHECKPOINT_ARTIFACT_ROOT/ema-serving-dcp-transfer-manifest.json"
export CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT=/mnt/workspace/cwam_cosmos3_robocasa365_eval/control/receipts/ema-checkpoint-atomic-publication.json

export TRAINING_CONFIG="$INPUTS_ROOT/config/R14-config.yaml"
export ACTION_STATS="$INPUTS_ROOT/config/robocasa365_native_frame_wise_relative_rot6d.json"
export WAN2PT2_VAE="$INPUTS_ROOT/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
export QWEN_TOKENIZER="$INPUTS_ROOT/models/Qwen3-VL-8B-Instruct-tokenizer"
export QWEN_TOKENIZER_MANIFEST="$INPUTS_ROOT/config/Qwen3-VL-8B-Instruct-tokenizer-manifest.json"
export ASSET_ENV_CLOSURE_MANIFEST=<copied-closure-manifest>
export ASSET_ENV_CLOSURE_H_VERIFICATION=<copied-h-verification-receipt>

export EXPECTED_H_SERVING_CHECKPOINT_MANIFEST_SHA256=7bcb9494b9f00b93dbad7d2b533032d3c8086e134986ae70f1c03abb1858f500
export EXPECTED_SERVING_CHECKPOINT_TREE_SHA256=77d2a84a35e81438c87a6e31bedc09299b294fcbe9841f4e943b985bbcb6f846
export EXPECTED_EMA_EXPORT_RECEIPT_SHA256=26bc5f658aa71a2ac8018dabb53574c9917171bb69bff9e4091d5621961c4200
export EXPECTED_EMA_KEY_MANIFEST_SHA256=e2edc73c06b628bb9294b157af1f6d3be3ec11c626713a709c9f64667ca2d8fa
export EXPECTED_EMA_VALIDATION_SHA256=887d7a962bda3f37f105d210945ec21bebed4becb0c42677b8c4ead256b673dd
export EXPECTED_EMA_SOURCE_DCP_FULL_RECEIPT_SHA256=1f9352ff4d97116d4861e2fec6dc5522a738a5ac58d5e15bf0025f3b1f45f646
export EXPECTED_EMA_TRANSFER_MANIFEST_SHA256=1f00b9d45f1d7ca2e6691fcd117d9da42c3444f9e517f5f3091362e1b3992f9b
export EXPECTED_CHECKPOINT_ATOMIC_PUBLICATION_RECEIPT_SHA256=<post-publication-receipt-sha256>
export EXPECTED_TRAINING_CONFIG_SHA256=c8269c4f12c99ac98e71e85334933d9749cd66a494960d6afe6e91368134caaa
export EXPECTED_ACTION_STATS_SHA256=2b48eb8a42e0c9c7befc45034d152282dad328f36389c10cafdc5be2635b67df
export EXPECTED_WAN2PT2_VAE_SHA256=20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36
export EXPECTED_QWEN_TOKENIZER_MANIFEST_SHA256=c71787b714a6876114b69d249e88bfa50de1262d33f46f8285dd00da729d4978
export EXPECTED_ASSET_ENV_CLOSURE_MANIFEST_SHA256=08b73f93a2ef191e2d82783e0a35689d1416a6d05e0384145e78551b3103f5bb
export EXPECTED_ASSET_ENV_CLOSURE_H_VERIFICATION_SHA256=60263e2de95efeeab89bc04b0b176331440a46ef8f027c8d708f8a9ecf739d45
export EGL_MAPPING_RECEIPT="$INPUTS_ROOT/config/robocasa-egl-device-mapping-dsw974.json"
export EXPECTED_EGL_MAPPING_RECEIPT_SHA256=88c1b81ea58fae409a0578956ac9c8114effccba85d644bebe72fdc8aad1a066
export EXPECTED_EGL_VENDOR_JSON_SHA256=9e6f14afaa7523370ad96e6675d3357441ee3985e60405cb3e0ced30b2ddaf76

export DSW_CPFS_MOUNT=/mnt/workspace
export DSW_CPFS_OUTPUT_PARENT=/mnt/workspace/cwam_cosmos3_robocasa365_eval/runs
export DSW_LOCAL_SCRATCH_PARENT=/tmp/cwam_cosmos3_robocasa365_eval
export EVAL_EXPERIMENT_ID=<new-formal-experiment-id>
export EVAL_LOGICAL_RUN_ID=<new-logical-run-id>
export EVAL_RUN_NAME=<new-run-name>
export EVAL_OUTPUT_ROOT="$DSW_CPFS_OUTPUT_PARENT/$EVAL_EXPERIMENT_ID/$EVAL_LOGICAL_RUN_ID"
export EVAL_EGL_VENDOR_JSON=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

bash "$COSMOS_SOURCE/examples/aliyun_4090/run_cwam_cosmos3_robocasa365_eval_dsw.sh"
```

## Watchdogs and terminal evidence

The torchrun server inherits a pinned NCCL watchdog environment, including
async error handling, monitoring, desync diagnostics, timeout dumps, a
900-second heartbeat timeout, and a trace buffer. The runtime receipt reads the
rank-0 process environment from `/proc` and rejects a missing or changed value.
Server readiness, the preflight episode, every evaluator shard, controlled
shutdown, and torchrun exit all have finite timeouts.

Torchrun starts as the leader of an isolated session/process group. Failure
cleanup signals the complete group, including all four ranks, before removing
the scoped node-local scratch directory.

After all eight formal shards finish, the client sends the authenticated
launch-nonce shutdown control message. A formal terminal receipt is produced
only if the server acknowledges it, torchrun exits with code zero, and the
immutable final server log is hashed. An abnormal trap may terminate the
process tree, but that path cannot produce passing terminal evidence.

The evaluation is a completed formal result only when all of these artifacts
exist and `control/sealed-evidence-verification.json` reports
`status="passed"`:

- `summary.csv` with exactly 340 independently reverified episode rows;
- `task_success_rates.csv` with exactly 17 complete task rows;
- `evaluation_evidence.json`, binding raw results, runtime, terminal, model,
  config, stats, VAE, tokenizer, source, assets, environments, and the separate
  one-episode gate; and
- `control/sealed-evidence-verification.json`, with one physical server, four
  physical GPUs, and eight logical shards.

Server readiness, loss values, a successful preflight episode, or a healthy
checkpoint alone are not a final RoboCasa365 control-success result.
