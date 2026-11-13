# Allostery fork of Pythia-PPI

This fork lets `hotspots` run Pythia-PPI on machines without a graphics processing unit (GPU). `hotspots` holds it as a submodule at `vendor/pythia-ppi`.

| Item | Value |
|---|---|
| Upstream | `https://github.com/Wublab/pythia_ppi` |
| Upstream commit | `1ec7b85df9cfbd86e6e68828d384852b232c5e22`, pushed 2026-06-13 |
| Licence | Apache-2.0, see `LICENSE` |

## What this fork changes

Two lines, both about where tensors live. Neither changes a number the model reads.

| File and function | Upstream | This fork | Why |
|---|---|---|---|
| `utils/pythia/pdb_utils.py`, `get_neighbor` | `device = "cuda"` | `device = protbb.ca.device` | A local variable, so no caller can override it. Any machine without CUDA fails there. |
| `utils/dataset.py`, `inference_process` | calls `inference_extarct_feature` without `device` | passes `device=device` | The callee fell back to its own `"cuda"` default, whatever device the caller asked for. |

On a GPU machine both lines behave as upstream does. The training code paths still default to `"cuda"`, because `hotspots` does not run them yet.

## Upstream faults kept on purpose

`get_neighbor` fills each node's torsion block with the central residue's own angles. It repeats them across all 32 neighbour slots.

`read_pdb_to_protbb` passes its atom lists to `ProtBB` in the wrong order. Nitrogen and oxygen are therefore swapped in every distance the model reads.

Both faults were present when the published weights were fitted, so the weights expect them. Fixing either without retraining would feed the network features it never saw. Predictions would get worse with nothing to show it. Fix them only in the same change that retrains.

`save_all` in `pdb_utils.py` calls `Parallel`, `delayed` and `tqdm` without importing them, so it raises if called. `hotspots` never calls it.

## Taking a newer upstream commit

Merge it into this fork, then run the `hotspots` reference tests against the merge. Move the submodule pin in `hotspots` only after they pass. Those tests compare the `hotspots` featurizer with this repository's own `inference_process`.
