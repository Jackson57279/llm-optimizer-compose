"""Streaming benchmark client — TTFT, TPS, ITL per prompt.

Drives any OpenAI-compatible endpoint (LiteLLM proxy, engine_manager directly,
or a raw llama-server). Measures:
    - TTFT: wall time to first token
    - ITL: inter-token latency list
    - TPS: completion_tokens / gen_duration

Writes a single JSON artifact per run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:
    print("httpx not installed; run `uv pip install -e .[server]`", file=sys.stderr)
    raise


def stream_one(
    client: httpx.Client,
    target: str,
    api_key: str,
    alias: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    url = target.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": alias,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
    }

    started = time.perf_counter()
    first_ts: float | None = None
    last_ts = started
    itl: list[float] = []
    tokens = 0
    text_parts: list[str] = []
    err: str | None = None

    try:
        with client.stream("POST", url, headers=headers, json=body, timeout=900.0) as resp:
            if resp.status_code != 200:
                err = f"HTTP {resp.status_code}: {resp.read().decode('utf-8', 'ignore')[:400]}"
                resp.close()
                raise RuntimeError(err)
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    msg = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = msg.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    now = time.perf_counter()
                    if first_ts is None:
                        first_ts = now
                    else:
                        itl.append(now - last_ts)
                    last_ts = now
                    tokens += 1
                    text_parts.append(content)
    except Exception as e:  # noqa: BLE001
        err = err or str(e)

    ended = time.perf_counter()
    ttft = (first_ts - started) if first_ts is not None else None
    gen_duration = (ended - first_ts) if first_ts is not None else None
    tps = (tokens / gen_duration) if (gen_duration and gen_duration > 0) else None

    return {
        "prompt_tokens_est": len(prompt.split()),
        "completion_tokens": tokens,
        "ttft_s": ttft,
        "gen_duration_s": gen_duration,
        "total_duration_s": ended - started,
        "tps": tps,
        "itl_mean_s": statistics.mean(itl) if itl else None,
        "itl_p95_s": (statistics.quantiles(itl, n=20)[18] if len(itl) >= 20 else None),
        "text": "".join(text_parts),
        "error": err,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="miniforge streaming bench client")
    ap.add_argument("--alias", required=True)
    ap.add_argument("--target", default="http://127.0.0.1:4000")
    ap.add_argument("--api-key", default="sk-local-dev-key")
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--default-max-tokens", type=int, default=128)
    args = ap.parse_args()

    prompts = []
    with args.prompts.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    results: list[dict[str, Any]] = []
    with httpx.Client() as client:
        for i, row in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] {row['id']}...", flush=True)
            t0 = time.perf_counter()
            r = stream_one(
                client,
                args.target,
                args.api_key,
                args.alias,
                row["prompt"],
                int(row.get("max_tokens", args.default_max_tokens)),
            )
            r["prompt_id"] = row["id"]
            r["wall_s"] = time.perf_counter() - t0
            results.append(r)
            if r.get("error"):
                print(f"      ERROR: {r['error']}")
            else:
                print(
                    f"      ttft={r['ttft_s']:.2f}s tps={r['tps']:.2f} tok={r['completion_tokens']}"
                    if r.get("tps")
                    else f"      done in {r['total_duration_s']:.1f}s (no tps)"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # compute simple aggregates for convenience
    ok = [r for r in results if not r.get("error") and r.get("tps")]
    aggregate = {
        "alias": args.alias,
        "target": args.target,
        "timestamp": time.time(),
        "n_prompts": len(results),
        "n_ok": len(ok),
        "tps_mean": statistics.mean([r["tps"] for r in ok]) if ok else None,
        "tps_median": statistics.median([r["tps"] for r in ok]) if ok else None,
        "ttft_mean_s": statistics.mean([r["ttft_s"] for r in ok if r.get("ttft_s")]) if ok else None,
        "ttft_p95_s": (
            statistics.quantiles([r["ttft_s"] for r in ok if r.get("ttft_s")], n=20)[18]
            if len(ok) >= 20
            else (
                max(r["ttft_s"] for r in ok if r.get("ttft_s")) if ok else None
            )
        ),
        "total_completion_tokens": sum(r.get("completion_tokens", 0) for r in ok),
    }
    args.output.write_text(
        json.dumps({"aggregate": aggregate, "runs": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {args.output}")
    if aggregate["n_ok"]:
        print(
            f"  mean TPS: {aggregate['tps_mean']:.2f}   median TPS: {aggregate['tps_median']:.2f}"
        )
    return 0 if aggregate["n_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
