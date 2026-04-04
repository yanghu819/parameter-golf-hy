#!/usr/bin/env python3
from __future__ import annotations

import io
import lzma
import math
import os
import sys
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import train_gpt as pg  # noqa: E402


def build_model(args: pg.Hyperparameters, device: torch.device) -> pg.GPT:
    model = pg.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        gated_attention=args.gated_attention,
        value_residual=args.value_residual,
    ).to(device).bfloat16()
    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    for module in model.modules():
        if isinstance(module, pg.CastedLinear):
            module.float()
    pg.restore_low_dim_params_to_fp32(model)
    return model


def encode_model(model: pg.GPT, input_ids: Tensor) -> Tensor:
    n = model.num_layers
    x = model.tok_emb(input_ids)
    if model.bigram is not None:
        x = x + model.bigram(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x = model.smear(x)
    x0 = x
    v0 = None
    skips: list[Tensor] = []
    ve_cache: dict[str, Tensor] = {}
    for i in range(model.num_encoder_layers):
        ve = model._get_ve(i, input_ids, ve_cache)
        x, raw_v = model.blocks[i](
            x,
            x0,
            model.qo_bank[i],
            model.kv_bank[i],
            model.kv_bank[n + i],
            model.qo_bank[n + i],
            model.mlp_up_bank[i],
            model.mlp_down_bank[i],
            v_embed=ve,
            v0=v0,
        )
        if v0 is None and raw_v is not None:
            v0 = raw_v
        skips.append(x)
    for i in range(model.num_decoder_layers):
        bi = model.num_encoder_layers + i
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        ve = model._get_ve(bi, input_ids, ve_cache)
        x, _ = model.blocks[bi](
            x,
            x0,
            model.qo_bank[bi],
            model.kv_bank[bi],
            model.kv_bank[n + bi],
            model.qo_bank[n + bi],
            model.mlp_up_bank[bi],
            model.mlp_down_bank[bi],
            v_embed=ve,
            v0=v0,
        )
    return model.final_norm(x)


def logits_from_hidden(model: pg.GPT, hidden: Tensor) -> Tensor:
    if model.tie_embeddings:
        logits_proj = F.linear(hidden, model.tok_emb.weight)
    else:
        if model.lm_head is None:
            raise RuntimeError("lm_head is required when tie_embeddings=False")
        logits_proj = model.lm_head(hidden)
    return model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)


def loss_from_hidden(
    model: pg.GPT,
    hidden: Tensor,
    target_ids: Tensor,
    hidden_delta: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    if hidden_delta is not None:
        hidden = hidden + hidden_delta.to(dtype=hidden.dtype)[:, None, :]
    logits = logits_from_hidden(model, hidden)
    if reduction == "none":
        bsz, seqlen, vocab = logits.shape
        return F.cross_entropy(
            logits.float().reshape(-1, vocab),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape(bsz, seqlen)
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction=reduction)


def load_checkpoint_into_model(model: pg.GPT, ckpt_path: Path, args: pg.Hyperparameters) -> str:
    if ckpt_path.suffix == ".ptz":
        quant_state = torch.load(io.BytesIO(lzma.decompress(ckpt_path.read_bytes())), map_location="cpu")
        template_bank = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        template_unbanked = pg._unbank_state_dict(template_bank, args.num_layers)
        deq_unbanked = pg.dequantize_mixed_int6(quant_state["w"], quant_state["m"], template_unbanked)
        state_dict = pg._rebank_state_dict(deq_unbanked, args.num_layers, template_bank)
        model.load_state_dict(state_dict, strict=True)
        return "int6_lzma"

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return "fp"


def find_docs(all_tokens: Tensor, bos_id: int) -> list[tuple[int, int]]:
    bos_positions = (all_tokens == bos_id).nonzero(as_tuple=True)[0].tolist()
    docs: list[tuple[int, int]] = []
    for i, start in enumerate(bos_positions):
        end = bos_positions[i + 1] if i + 1 < len(bos_positions) else int(all_tokens.numel())
        if end - start >= 2:
            docs.append((int(start), int(end - start)))
    return docs


def compute_chunk_window(ci: int, pred_len: int, num_chunks: int, chunk_size: int, eval_seq_len: int) -> tuple[int, int, int, int]:
    chunk_start = ci * chunk_size
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start
    return win_start, win_len, chunk_offset, chunk_len


def accumulate_bpb(
    ptl: Tensor,
    x: Tensor,
    y: Tensor,
    batch_i: int,
    chunk_offset: int,
    chunk_len: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    loss_sum: Tensor,
    byte_sum: Tensor,
    token_count: Tensor,
) -> None:
    lbl = ptl[batch_i, chunk_offset:chunk_offset + chunk_len].to(torch.float64)
    prev = x[batch_i, chunk_offset:chunk_offset + chunk_len]
    tgt = y[batch_i, chunk_offset:chunk_offset + chunk_len]
    tok_bytes = base_bytes_lut[tgt].to(torch.float64)
    tok_bytes += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
    loss_sum += lbl.sum()
    byte_sum += tok_bytes.sum()
    token_count += chunk_len


def eval_val_slot_delta(
    args: pg.Hyperparameters,
    model: pg.GPT,
    device: torch.device,
    all_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    bos_id: int,
) -> tuple[float, float]:
    chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 32))
    eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 1024))
    batch_size = int(os.environ.get("TTT_BATCH_SIZE", 16))
    max_docs = int(os.environ.get("TTT_MAX_DOCS", 0))
    slot_lr = float(os.environ.get("TTT_SLOT_LR", 3e-4))
    slot_steps = int(os.environ.get("TTT_SLOT_STEPS", 1))
    adapt_block = max(1, int(os.environ.get("TTT_SLOT_ADAPT_BLOCK", 1)))
    slot_eps = float(os.environ.get("TTT_SLOT_EPS", 1e-5))
    slot_weight_decay = float(os.environ.get("TTT_SLOT_WEIGHT_DECAY", 1e-8))
    slot_delta_clip = float(os.environ.get("TTT_SLOT_DELTA_CLIP", 10.0))
    slot_update = os.environ.get("TTT_SLOT_UPDATE", "adamw").strip().lower()
    slot_update_every = max(1, int(os.environ.get("TTT_SLOT_UPDATE_EVERY", 2)))
    slot_update_first_k_chunks = max(0, int(os.environ.get("TTT_SLOT_UPDATE_FIRST_K_CHUNKS", 0)))
    slot_update_loss_threshold = float(os.environ.get("TTT_SLOT_UPDATE_LOSS_THRESHOLD", 0.0))

    if slot_update not in {"adamw", "sgd", "norm"}:
        raise ValueError(f"Unsupported TTT_SLOT_UPDATE={slot_update!r}")

    docs = find_docs(all_tokens, bos_id)
    if max_docs > 0:
        docs = docs[:max_docs]
    docs.sort(key=lambda d: (d[1] - 2 + chunk_size - 1) // chunk_size)

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)

    autocast_enabled = device.type == "cuda"

    for bi in range(0, len(docs), batch_size):
        batch = docs[bi:bi + batch_size]
        bsz = len(batch)
        pred_lens = [doc_len - 1 for _, doc_len in batch]
        num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc = max(num_chunks)

        delta = nn.Parameter(torch.zeros(bsz, args.model_dim, device=device, dtype=torch.float32))
        optimizer = None
        if slot_steps > 0 and slot_update == "adamw":
            optimizer = torch.optim.AdamW(
                [delta],
                lr=slot_lr,
                betas=(args.beta1, args.beta2),
                eps=slot_eps,
                weight_decay=slot_weight_decay,
            )

        for ci in range(max_nc):
            _, context_size, _, _ = compute_chunk_window(ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len)
            active = [ci < nc for nc in num_chunks]
            needs_train = any(ci < nc - 1 for nc in num_chunks)

            x = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            doc_info: list[tuple[int, int]] = []
            for b, (doc_start, doc_len) in enumerate(batch):
                if not active[b]:
                    doc_info.append((0, 0))
                    continue
                ws, wl, co, cl = compute_chunk_window(ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len)
                toks = all_tokens[doc_start + ws: doc_start + ws + wl + 1].to(dtype=torch.int64, device=device)
                x[b, :wl] = toks[:-1]
                y[b, :wl] = toks[1:]
                doc_info.append((co, cl))

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                hidden = encode_model(model, x)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                ptl = loss_from_hidden(model, hidden, y, hidden_delta=delta, reduction="none")

            with torch.no_grad():
                chunk_loss_total = torch.zeros((), device=device, dtype=torch.float64)
                chunk_loss_tokens = torch.zeros((), device=device, dtype=torch.float64)
                for b, (co, cl) in enumerate(doc_info):
                    if not active[b]:
                        continue
                    chunk_loss_total += ptl[b, co:co + cl].sum(dtype=torch.float64)
                    chunk_loss_tokens += float(cl)
                    accumulate_bpb(
                        ptl,
                        x,
                        y,
                        b,
                        co,
                        cl,
                        base_bytes_lut,
                        has_leading_space_lut,
                        is_boundary_token_lut,
                        loss_sum,
                        byte_sum,
                        token_count,
                    )

            chunk_mean_loss = float((chunk_loss_total / chunk_loss_tokens.clamp_min(1.0)).item())
            if not needs_train or slot_steps <= 0 or ((ci + 1) % slot_update_every) != 0:
                continue
            if slot_update_first_k_chunks > 0 and ci >= slot_update_first_k_chunks:
                continue
            if slot_update_loss_threshold > 0.0 and chunk_mean_loss <= slot_update_loss_threshold:
                continue

            active_mask = torch.tensor([float(ci < num_chunks[b] - 1) for b in range(bsz)], device=device, dtype=torch.float32)
            for micro_start in range(0, chunk_size, adapt_block):
                micro_end = min(micro_start + adapt_block, chunk_size)
                if not torch.any(active_mask):
                    break
                for _ in range(slot_steps):
                    if delta.grad is not None:
                        delta.grad = None
                    if optimizer is not None:
                        optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                        loss = torch.zeros((), device=device, dtype=torch.float32)
                        contrib = 0
                        for b, (co, cl) in enumerate(doc_info):
                            if active_mask[b].item() == 0.0 or micro_start >= cl:
                                continue
                            ms = co + micro_start
                            me = co + min(micro_end, cl)
                            if me <= ms:
                                continue
                            block_loss = loss_from_hidden(
                                model,
                                hidden[b:b + 1, ms:me, :],
                                y[b:b + 1, ms:me],
                                hidden_delta=delta[b:b + 1],
                                reduction="none",
                            ).mean()
                            loss = loss + block_loss
                            contrib += 1
                    if contrib == 0:
                        continue
                    loss.backward()
                    if slot_update == "adamw":
                        assert optimizer is not None
                        optimizer.step()
                    else:
                        with torch.no_grad():
                            grad = delta.grad
                            if grad is None:
                                continue
                            grad = grad.to(dtype=torch.float32)
                            if slot_update == "norm":
                                grad_norm = grad.norm(dim=-1, keepdim=True).clamp_min(slot_eps)
                                grad = grad / grad_norm
                            if slot_weight_decay != 0.0:
                                delta.mul_(1.0 - slot_lr * slot_weight_decay)
                            delta.add_(grad, alpha=-slot_lr)
                    if slot_delta_clip > 0.0:
                        with torch.no_grad():
                            norms = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                            scale = (slot_delta_clip / norms).clamp_max(1.0)
                            delta.mul_(scale)

    val_loss = float(loss_sum.item() / token_count.item())
    val_bpb = float((loss_sum.item() / math.log(2.0)) / byte_sum.item())
    model.train()
    return val_loss, val_bpb


def main() -> int:
    ckpt = os.environ.get("CHECKPOINT_PATH", "").strip()
    if not ckpt:
        raise SystemExit("CHECKPOINT_PATH is required")
    ckpt_path = Path(ckpt).resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    args = pg.Hyperparameters()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is required for cheap parity with the record stack")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    pg.random.seed(args.seed)
    pg.np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise SystemExit(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    bos_id = int(sp.bos_id()) if int(sp.bos_id()) >= 0 else 1

    files = [Path(p) for p in sorted(pg.glob.glob(args.val_files))]
    if not files:
        raise SystemExit(f"No val files found for pattern: {args.val_files}")
    all_tokens = torch.cat([pg.load_data_shard(file) for file in files]).contiguous()
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = pg.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    model = build_model(args, device)
    checkpoint_kind = load_checkpoint_into_model(model, ckpt_path, args)

    val_loss, val_bpb = eval_val_slot_delta(
        args,
        model,
        device,
        all_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        bos_id,
    )

    print(f"checkpoint:{ckpt_path}")
    print(f"checkpoint_kind:{checkpoint_kind}")
    print("ttt_mode:slot_delta")
    print(f"proxy_docs:{int(os.environ.get('TTT_MAX_DOCS', '0'))}")
    print(f"ttt_chunk_size:{int(os.environ.get('TTT_CHUNK_SIZE', 32))}")
    print(f"ttt_eval_seq_len:{int(os.environ.get('TTT_EVAL_SEQ_LEN', 1024))}")
    print(f"ttt_batch_size:{int(os.environ.get('TTT_BATCH_SIZE', 16))}")
    print(f"ttt_slot_steps:{int(os.environ.get('TTT_SLOT_STEPS', 1))}")
    print(f"ttt_slot_adapt_block:{int(os.environ.get('TTT_SLOT_ADAPT_BLOCK', 1))}")
    print(f"ttt_slot_lr:{float(os.environ.get('TTT_SLOT_LR', 3e-4))}")
    print(f"ttt_slot_update_every:{int(os.environ.get('TTT_SLOT_UPDATE_EVERY', 2))}")
    print(f"val_loss:{val_loss:.8f}")
    print(f"val_bpb:{val_bpb:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
