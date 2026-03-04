# minAlphaFold2 input/output pipeline (mapped to AlphaFold2 paper)

This document gives a practical "where to look first" walkthrough from **model inputs** to **model outputs** and loss terms, with references to AlphaFold2 Supplement algorithms.

## 1) Top-level model entry point

Main entry: `AlphaFold2.forward(...)` in `minalphafold/model.py`.

### Forward inputs

- `target_feat` `(B, N_res, 21)` — per-residue target sequence features.
- `residue_index` `(B, N_res)` — residue indices used by relative positional encoding.
- `msa_feat` `(B, N_seq, N_res, 49)` — clustered MSA features.
- `extra_msa_feat` `(B, N_extra, N_res, 25)` — extra MSA features.
- `template_pair_feat` `(B, N_templ, N_res, N_res, 88)` — template pairwise features.
- `aatype` `(B, N_res)` — amino-acid identities for structure generation.
- Optional: `template_angle_feat`, `template_mask`, `seq_mask`, `msa_mask`, `extra_msa_mask`.
- Control params: `n_cycles` (recycling), `n_ensemble` (ensemble averaging).

## 2) Input embedding stage (Suppl. Alg. 3 + 4)

Implemented in `InputEmbedder` (`minalphafold/embedders.py`):

- Creates initial **pair representation** `z` by combining projected target features for row/column residues.
- Adds **relative positional encoding** from `RelPos` based on `residue_index`.
- Creates initial **MSA representation** `m` by combining projected target and MSA features.

Outputs:

- `m`: `(B, N_seq, N_res, c_m)`
- `z`: `(B, N_res, N_res, c_z)`

## 3) Recycling setup (Suppl. Alg. 30/31/32)

In `model.py`, before iterative cycles:

- Initializes previous-cycle tensors (`single_rep_prev`, `z_prev`, `x_prev`) to zeros.
- For each cycle:
  - Adds normalized recycled single representation to first MSA row.
  - Adds normalized recycled pair representation.
  - Adds learned projection of distance bins computed from previous coordinates (`recycling_distance_bin`).

Training mode follows Algorithm 31 behavior: samples cycle count uniformly in `[1, n_cycles]`.

## 4) Template pipeline (Suppl. Alg. 16 + 17)

In `model.py` + `embedders.py`:

1. Project `template_pair_feat` to `c_t` then run `TemplatePair` stack (triangle ops + pair transition).
2. Fuse template information into current pair representation with `TemplatePointwiseAttention`.
3. If `template_angle_feat` is provided, project it to `c_m` and append as extra MSA rows.

## 5) Extra MSA pipeline (Suppl. Alg. 18 + 19)

In `model.py` + `ExtraMsaStack` (`embedders.py`):

- Project `extra_msa_feat` to `c_e`.
- Run multiple extra-MSA blocks, each doing:
  - MSA row attention with pair bias,
  - MSA column global attention,
  - MSA transition,
  - outer-product mean update to pair,
  - triangle multiplicative/attention pair updates,
  - pair transition.

Output after stack: updated pair representation (and extra-MSA latent state used internally).

## 6) Evoformer trunk (Suppl. Alg. 6–15)

In `model.py` + `minalphafold/evoformer.py`:

- Run `num_evoformer` blocks over `(msa_repr, pair_repr)`.
- Each Evoformer block includes:
  - MSA row attention with pair bias,
  - MSA column attention,
  - MSA transition,
  - outer-product mean,
  - triangle multiplication (incoming/outgoing),
  - triangle attention (starting/ending node),
  - pair transition.

After all blocks:

- First MSA row becomes "single" token stream (`msa_first_row`) and is projected to `c_s` as `single_rep`.

## 7) Structure module (Suppl. Alg. 20, 22–25)

`StructureModule` (`minalphafold/structure_module.py`) consumes:

- `single_rep` `(B, N_res, c_s)`
- `pair_repr` `(B, N_res, N_res, c_z)`
- `aatype` `(B, N_res)`

Key operations:

- Iterative IPA + transition + backbone updates over `structure_module_layers`.
- Predicts residue frames (rotations/translations), torsions, and atom14 coordinates.
- Maintains trajectories over layers (`traj_*`) and final structure outputs.

Core structure outputs include:

- `atom14_coords`, `atom14_mask`
- `final_rotations`, `final_translations`
- `all_frames_R`, `all_frames_t`
- `traj_rotations`, `traj_translations`, `traj_torsion_angles`
- updated per-residue single representation (`single`)

## 8) Prediction heads (Suppl. Alg. 29 and AF2 heads)

On the last recycle iteration (`model.py`):

- `DistogramHead(pair_repr)` → `distogram_logits`
- `MaskedMSAHead(msa_repr)` → `masked_msa_logits`
- `ExperimentallyResolvedHead(single_rep)` → `experimentally_resolved_logits`
- `PLDDTHead(structure_predictions["single"])` → `plddt_logits`
- `TMScoreHead(pair_repr)` → `tm_logits`

Returned dictionary also includes latent reps:

- `pair_representation`, `msa_representation`, `single_representation`

## 9) Recycling handoff between cycles

For non-final cycles in `model.py`:

- Stores detached `msa_first_row` and `pair_repr` as next-cycle recycled reps.
- Extracts pseudo-beta positions from predicted `atom14_coords` (CA for glycine, CB otherwise) for next-cycle distance-bin features.

## 10) Training losses (paper alignment)

Implemented in `minalphafold/losses.py` via `AlphaFoldLoss`:

- FAPE (backbone + all-atom)
- torsion-angle loss
- pLDDT loss
- distogram loss
- masked-MSA loss
- structural-violation losses

This codebase includes head modules and loss modules, but dataset preprocessing pieces for some supplement algorithms (e.g., MSA block deletion, rigid-frame GT construction) are marked as not implemented in README.

## 11) Practical “read order” to map to the paper

1. `minalphafold/model.py` (`AlphaFold2.forward`) for full control flow.
2. `minalphafold/embedders.py` (`InputEmbedder`, template + extra-MSA modules).
3. `minalphafold/evoformer.py` (core trunk block).
4. `minalphafold/structure_module.py` (coordinate generation).
5. `minalphafold/heads.py` and `minalphafold/losses.py` (training objectives and confidence outputs).
6. `README.md` supplement-algorithm table for quick cross-reference.
