# Sources inventory

The `sources/` directory is **excluded from this archive** because it is large
(3.4 GB) and consists mostly of upstream checkouts and binary evidence.
This file records what each directory contains and where it came from, so the
tree can be restored.

| Path | Size | Files | Origin | Revision |
|---|---:|---:|---|---|
| `sources/cudnn_frontend/` | 233M | 5750 | https://github.com/NVIDIA/cudnn-frontend.git | `develop` @ `a736a06` |
| `sources/cutlass/` | 238M | 8153 | https://github.com/NVIDIA/cutlass.git | `main` @ `b2f249a` |
| `sources/keras/` | 18M | 1097 | https://github.com/keras-team/keras.git | `master` @ `2818e6c` |
| `sources/llvm/` | 120M | 76 | https://github.com/llvm/llvm-project.git | partial clone (`blob:none`), `main` |
| `sources/pytorch/` | 372M | 23544 | https://github.com/pytorch/pytorch.git | `main` @ `30c5b5d` |
| `sources/tensorflow/` | 788M | 37911 | https://github.com/tensorflow/tensorflow.git | `master` @ `a42d246` |
| `sources/decompiled/` | 18M | 42 | Local SM75 decompilation output | n/a |
| `sources/gold_verified/` | 12K | 2 | Hand-verified cases | n/a |
| `sources/server-kernel-evidence/` | 1.7G | 481 | Local GPU evidence bundle | n/a |

## Local (non-git) directories

- `sources/decompiled/sm75/` — `manifests/`, `host_dispatch/`, `functions/`.
- `sources/gold_verified/` — `pytorch_conv_bn_eval.jsonl`,
  `pytorch_inductor_epilogue_sm75.jsonl`.
- `sources/server-kernel-evidence/` — see its own `README.md`; contains
  `artifacts/`, `external/`, `sm75-demangled/`, `sm75-source-map/`.

## Restoring upstream checkouts

```bash
mkdir -p sources
git clone https://github.com/NVIDIA/cudnn-frontend.git sources/cudnn_frontend
git -C sources/cudnn_frontend checkout a736a06
git clone https://github.com/NVIDIA/cutlass.git sources/cutlass
git -C sources/cutlass checkout b2f249a
git clone https://github.com/keras-team/keras.git sources/keras
git -C sources/keras checkout 2818e6c
git clone --filter=blob:none https://github.com/llvm/llvm-project.git sources/llvm
git clone https://github.com/pytorch/pytorch.git sources/pytorch
git -C sources/pytorch checkout 30c5b5d
git clone https://github.com/tensorflow/tensorflow.git sources/tensorflow
git -C sources/tensorflow checkout a42d246
```

The remaining local directories (`decompiled/`, `gold_verified/`,
`server-kernel-evidence/`) have no upstream; copy them from the original
machine or regenerate them with the project tooling.
