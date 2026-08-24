#!/usr/bin/env python3
"""Production-shape FP64 oracle measurements for one FlashMLA binary/policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


PAGE = 64
CACHE_TOKENS = 65536
TOPK = 2048
PADDED_HEADS = 64
ACTIVE_HEADS = 8
D_QK = 576
D_V = 512
TOKEN_BYTES = 656


def digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()


def configure(policy: str) -> None:
    if policy == "fast-combine":
        os.environ["WS58_FLASHMLA_POLICY"] = "active-combine-only"
        os.environ["WS58_FLASHMLA_FAST_SMALL_SPLITS"] = "1"
    else:
        os.environ["WS58_FLASHMLA_POLICY"] = policy
        os.environ["WS58_FLASHMLA_FAST_SMALL_SPLITS"] = "0"
    os.environ["WS58_FLASHMLA_ACTIVE_HEADS"] = str(ACTIVE_HEADS)
    os.environ["WS58_FLASHMLA_FIXED_CAP"] = "132"
    os.environ["WS58_C04_MAX_SM_PARTS"] = "132"


def one_seed(seed: int, policy: str) -> dict[str, object]:
    configure(policy)
    from vllm.third_party.flashmla.flash_mla_interface import (
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    blocks = CACHE_TOKENS // PAGE
    storage = torch.empty((blocks, PAGE, TOKEN_BYTES), dtype=torch.uint8, device="cuda")
    nope_fp8 = (
        torch.randn((blocks, PAGE, D_V), generator=generator, device="cuda")
        .mul_(0.1)
        .to(torch.float8_e4m3fn)
    )
    rope_bf16 = (
        torch.randn(
            (blocks, PAGE, D_QK - D_V), generator=generator, device="cuda"
        )
        .mul_(0.1)
        .to(torch.bfloat16)
    )
    storage[:, :, :D_V].copy_(nope_fp8.view(torch.uint8))
    storage[:, :, D_V : D_V + 16].view(torch.float32).fill_(1.0)
    storage[:, :, D_V + 16 :].copy_(
        rope_bf16.view(torch.uint8).reshape(blocks, PAGE, 128)
    )
    kv = storage.view(blocks, PAGE, 1, TOKEN_BYTES)
    q = torch.zeros((1, 1, PADDED_HEADS, D_QK), dtype=torch.bfloat16, device="cuda")
    q[:, :, :ACTIVE_HEADS].copy_(
        torch.randn(
            (1, 1, ACTIVE_HEADS, D_QK), generator=generator, device="cuda"
        ).to(torch.bfloat16)
    )
    indices = torch.randperm(CACHE_TOKENS, generator=generator, device="cuda")[
        :TOPK
    ].to(torch.int32).view(1, 1, TOPK)

    selected_indices = indices.flatten().to(torch.int64)
    selected_nope = nope_fp8.view(-1, D_V)[selected_indices].float().double()
    selected_rope = rope_bf16.view(-1, D_QK - D_V)[selected_indices].double()
    selected_qk = torch.cat((selected_nope, selected_rope), dim=-1)
    scores = q[0, 0, :ACTIVE_HEADS].double() @ selected_qk.T
    scores.mul_(D_QK**-0.5)
    oracle_lse = torch.logsumexp(scores, dim=-1)
    oracle_out = torch.softmax(scores, dim=-1) @ selected_nope

    scheduler, _ = get_mla_metadata()

    def call() -> tuple[torch.Tensor, torch.Tensor]:
        return flash_mla_with_kvcache(
            q=q,
            k_cache=kv,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=D_V,
            tile_scheduler_metadata=scheduler,
            is_fp8_kvcache=True,
            indices=indices,
            softmax_scale=D_QK**-0.5,
        )

    eager_out, eager_lse = call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_lse = call()
    graph.replay()
    torch.cuda.synchronize()
    kept_out = graph_out[0, 0, :ACTIVE_HEADS]
    kept_lse = graph_lse[0, :ACTIVE_HEADS, 0]
    return {
        "seed": seed,
        "input_sha256": {"q": digest(q), "kv": digest(kv), "indices": digest(indices)},
        "out_sha256": digest(kept_out),
        "lse_sha256": digest(kept_lse),
        "out_dtype": str(kept_out.dtype),
        "lse_dtype": str(kept_lse.dtype),
        "eager_graph_out_exact": bool(
            torch.equal(eager_out[:, :, :ACTIVE_HEADS], graph_out[:, :, :ACTIVE_HEADS])
        ),
        "eager_graph_lse_exact": bool(
            torch.equal(eager_lse[:, :ACTIVE_HEADS], graph_lse[:, :ACTIVE_HEADS])
        ),
        "out_fp64_mae": float((kept_out.double() - oracle_out).abs().mean().item()),
        "out_fp64_max_abs": float(
            (kept_out.double() - oracle_out).abs().max().item()
        ),
        "lse_fp64_mae": float((kept_lse.double() - oracle_lse).abs().mean().item()),
        "lse_fp64_max_abs": float(
            (kept_lse.double() - oracle_lse).abs().max().item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("baseline", "fast-combine"), required=True)
    parser.add_argument(
        "--seeds",
        default="271,827,404052,20260813,20260814,20260815,20260816,20260817,20260818,20260819",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    cases = [one_seed(int(seed), args.policy) for seed in args.seeds.split(",")]
    payload = {
        "policy": args.policy,
        "shape": {
            "batch": 1,
            "seq_q": 1,
            "heads_padded": PADDED_HEADS,
            "heads_active": ACTIVE_HEADS,
            "d_qk": D_QK,
            "d_v": D_V,
            "topk": TOPK,
            "cache_tokens": CACHE_TOKENS,
            "kv_layout_bytes": TOKEN_BYTES,
        },
        "cases": cases,
        "aggregate": {
            field: sum(float(case[field]) for case in cases) / len(cases)
            for field in ("out_fp64_mae", "lse_fp64_mae")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
