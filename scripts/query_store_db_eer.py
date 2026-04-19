#!/usr/bin/env python3
"""Read EER-related scalar metrics from an mlop local ``store.db`` (SQLite).

The database is expected to have a ``num`` table: ``step``, ``key``, ``value``, ``time``.
Given an approximate training step, pick the **nearest** ``num.step`` that has at least one
metric whose name contains ``eer`` (case-insensitive).

MLOP often advances ``num.step`` between successive ``run.log`` calls in one validation pass
(e.g. variant EER, super-hard EER, then primary ``val/eer/voxceleb1_o``). By default this
script **merges** EER keys across consecutive ``num.step`` rows that each contain at least
one EER metric, so primary and pooled metrics show up together with variant metrics.

Example::

    python scripts/query_store_db_eer.py \\
        /home/kirill/ASV/.mlop/asv/ablation_study_b0_model/store.db --step 5000
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


_COMPACT_METRIC_KEYS = (
    "val/eer/pooled",
    "val/eer/pooled_hard",
    "val/variants/eer_codec__vorbis_16k",
    "val/variants/eer_codec__speex_16k",
    "val/variants/eer_codec__pcm_mulaw_8k",
    "val/variants/eer_codec__opus_16k",
    "val/variants/eer_codec__mp3_16k",
    "val/variants/eer_codec__g722_16k",
    "val/variants/eer_codec__flac_16k",
    "val/variants/eer_codec__adpcm_ima_wav_16k",
    "val/variants/eer_codec__aac_16k",
    "val/variants/eer_aug__additive_noise__RawBoostTransform",
    "val/variants/eer_aug__additive_noise__AddBackgroundNoise",
    "val/variants/eer_aug__additive_noise__AddColoredNoise",
    "val/variants/eer_aug__rir__ApplyImpulseResponse",
)


def _nearest_eer_step(conn: sqlite3.Connection, target_step: int) -> int | None:
    cur = conn.execute(
        """
        SELECT step
        FROM (
            SELECT DISTINCT step AS step
            FROM num
            WHERE lower(key) LIKE '%eer%'
        )
        ORDER BY abs(step - ?) ASC, step ASC
        LIMIT 1
        """,
        (target_step,),
    )
    row = cur.fetchone()
    return int(row[0]) if row else None


def _eer_metrics_at_step(
    conn: sqlite3.Connection, step: int
) -> list[tuple[str, float, int]]:
    cur = conn.execute(
        """
        SELECT key, value, time
        FROM num
        WHERE step = ? AND lower(key) LIKE '%eer%'
        ORDER BY key
        """,
        (step,),
    )
    return [(str(k), float(v), int(t)) for k, v, t in cur.fetchall()]


def _step_has_eer(conn: sqlite3.Connection, step: int) -> bool:
    cur = conn.execute(
        """
        SELECT 1
        FROM num
        WHERE step = ? AND lower(key) LIKE '%eer%'
        LIMIT 1
        """,
        (step,),
    )
    return cur.fetchone() is not None


def _eer_metric_keys_at_step(conn: sqlite3.Connection, step: int) -> set[str]:
    cur = conn.execute(
        """
        SELECT key
        FROM num
        WHERE step = ? AND lower(key) LIKE '%eer%'
        """,
        (step,),
    )
    return {str(r[0]) for r in cur.fetchall()}


def _contiguous_eer_step_cluster(
    conn: sqlite3.Connection,
    anchor: int,
    *,
    max_expand: int,
) -> list[int]:
    """Steps ``anchor .. high`` and ``low .. anchor`` while each boundary step has EER rows."""
    low = anchor
    for _ in range(max_expand):
        if low <= 0 or not _step_has_eer(conn, low - 1):
            break
        low -= 1
    high = anchor
    for _ in range(max_expand):
        nxt = high + 1
        if not _step_has_eer(conn, nxt):
            break
        keys_here = _eer_metric_keys_at_step(conn, high)
        keys_next = _eer_metric_keys_at_step(conn, nxt)
        # Trailing partial ``run.log`` bursts (same keys as before, fewer rows) should not
        # extend the cluster or they overwrite good metrics from the fuller step.
        if keys_next <= keys_here and len(keys_next) < len(keys_here):
            break
        high = nxt
    return list(range(low, high + 1))


def _merge_eer_metrics_across_steps(
    conn: sqlite3.Connection, steps: list[int]
) -> list[tuple[str, float, int]]:
    """Later ``num.step`` wins on duplicate keys (final ``run.log`` overwrites earlier)."""
    merged: dict[str, tuple[float, int]] = {}
    for s in sorted(steps):
        for key, value, t in _eer_metrics_at_step(conn, s):
            merged[key] = (value, t)
    return sorted((k, v, t) for k, (v, t) in merged.items())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "store_db",
        type=Path,
        help="Path to store.db (e.g. .../.mlop/asv/<run>/store.db)",
    )
    p.add_argument(
        "--step",
        type=int,
        required=True,
        help="Approximate global step; the step with EER metrics closest to this is used.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON: chosen_step, merged_steps, target_step, step_delta, metrics.",
    )
    p.add_argument(
        "--no-merge-consecutive",
        action="store_true",
        help="Only read the single nearest EER step (do not merge split validation logs).",
    )
    p.add_argument(
        "--max-cluster-expand",
        type=int,
        default=32,
        metavar="N",
        help="Max consecutive num.step rows to walk each direction when merging (default: 32).",
    )
    p.add_argument(
        "--compact-metrics",
        action="store_true",
        help=(
            "Print only selected metrics in order, separated by '&', with two decimal places."
        ),
    )
    args = p.parse_args()

    path = args.store_db.expanduser().resolve()
    if not path.is_file():
        print(f"error: not a file: {path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        chosen = _nearest_eer_step(conn, args.step)
        if chosen is None:
            print("No rows with 'eer' in metric key.", file=sys.stderr)
            sys.exit(2)
        if args.no_merge_consecutive:
            cluster = [chosen]
        else:
            cluster = _contiguous_eer_step_cluster(
                conn, chosen, max_expand=max(0, args.max_cluster_expand)
            )
        rows = _merge_eer_metrics_across_steps(conn, cluster)
    finally:
        conn.close()

    delta = chosen - args.step
    metrics = {k: v for k, v, _t in rows}

    if args.compact_metrics:
        compact_values = [
            f"{metrics[key]:.2f}" if key in metrics else "nan"
            for key in _COMPACT_METRIC_KEYS
        ]
        print("&".join(compact_values))
        return

    if args.json:
        out = {
            "target_step": args.step,
            "chosen_step": chosen,
            "step_delta": delta,
            "merged_steps": cluster,
            "metrics": metrics,
        }
        print(json.dumps(out, indent=2))
        return

    span = f"{cluster[0]}..{cluster[-1]}" if len(cluster) > 1 else str(cluster[0])
    print(
        f"target_step={args.step}  chosen_step={chosen}  merged_num_steps={span}  (delta {delta:+d})"
    )
    print(f"merged EER metrics ({len(rows)} keys):")
    for key, value, _t in rows:
        print(f"  {key}  {value}")


if __name__ == "__main__":
    main()
