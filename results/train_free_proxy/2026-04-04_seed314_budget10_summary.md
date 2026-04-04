## Train-Free Budget10 Search

- Checkpoint: `saved_runs/record_best_formal_seed314_20260404T085507Z/final_model.int6.ptz`
- Eval mode: `slot_delta`
- Fixed knobs: `chunk_size=32`, `eval_seq_len=1024`, `batch_size=16`
- Budget policy: cheap `1x RTX 4090` only

### Best So Far

| subset | baseline_bpb | best_setting | best_bpb | gain_bpb |
| --- | ---: | --- | ---: | ---: |
| docs32 | 1.44026872 | `adamw_best_lr5e4` | 1.43725812 | -0.00301060 |
| docs64 | 1.38872647 | `adamw_best_lr7e4` | 1.38637571 | -0.00235076 |
| docs128 | 1.38269022 | `adamw_best_lr7e4` | 1.38019458 | -0.00249564 |

### Takeaways

- `adamw`, `steps=1`, `adapt_block=1`, `update_every=1` stayed dominant throughout.
- The best learning rate for this checkpoint moved above the original `3e-4` slot-golf default; `7e-4` was best on both `docs64` and `docs128`.
- Sparse threshold schedules that looked promising in `slot-golf` did not transfer to this checkpoint. Both `thr=3.20` and `thr=3.30` nearly erased the gain on `docs32`.
- `gate2` remained weaker than `every=1` on this checkpoint.
