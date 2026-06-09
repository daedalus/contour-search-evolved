#!/usr/bin/env python3
"""AlphaEvolve runner: test candidates, benchmark, promote champion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alphaevolve.evaluator import evaluate_candidate, evaluate_candidate_detailed
from search.algorithms import contour_search as baseline


CANDIDATE_DIR = Path(__file__).parent / "candidates"


def load_candidate(path: Path):
    code = path.read_text()
    ns: dict[str, object] = {}
    exec(code, ns)
    fn = ns.get("contour_search")
    if fn is None:
        raise ValueError(f"No contour_search in {path.name}")
    return fn  # type: ignore[return-value]


def quick_correctness_check(fn) -> bool:
    """Run a quick smoke test: chain_5k, grid_2500, unreachable."""
    from search.graph import Graph

    # Chain 5k
    g = Graph()
    for i in range(5000):
        g.add_edge(str(i), str(i + 1), weight=1, bidirectional=True)

    def h(a, b):
        return abs(int(a) - int(b)) if a.isdigit() and b.isdigit() else 0.0

    result = fn(g, "0", "5000", h)
    if result is None or result[0] != "0" or result[-1] != "5000":
        return False

    # Grid 50x50 (sample)
    g2 = Graph()
    for r in range(10):
        for c in range(10):
            n = f"{r},{c}"
            if c + 1 < 10:
                g2.add_edge(n, f"{r},{c + 1}", weight=1, bidirectional=True)
            if r + 1 < 10:
                g2.add_edge(n, f"{r + 1},{c}", weight=1, bidirectional=True)

    def h2(a, b):
        return 0.0

    result2 = fn(g2, "0,0", "9,9", h2)
    if result2 is None:
        return False

    # Unreachable
    g3 = Graph()
    g3.add_edge("A", "B", weight=1, bidirectional=False)
    g3.add_edge("C", "D", weight=1, bidirectional=False)
    result3 = fn(g3, "A", "D", lambda a, b: 0.0)
    if result3 is not None:
        return False

    return True


def _print_baseline():
    print("=== Baseline ===")
    base_score = evaluate_candidate(baseline)
    base_details = evaluate_candidate_detailed(baseline)
    print(f"  Score: {base_score:.6f} (geo-mean: {1.0 / base_score * 1000:.4f}ms)")
    for name, d in base_details.items():
        if isinstance(d, dict) and "error" not in d:
            print(f"    {name}: median={d['median_ms']:.3f}ms  cv={d['cv']:.4f}")
    print()
    return base_score


def _load_candidates(candidate_filter):
    candidates = []
    for f in sorted(CANDIDATE_DIR.glob("*.py")):
        if candidate_filter and candidate_filter not in f.stem:
            continue
        try:
            fn = load_candidate(f)
            candidates.append((f.stem, fn))
            print(f"  Loaded {f.stem}")
        except Exception as e:
            print(f"  FAILED to load {f.stem}: {e}")
    return candidates


def _check_candidates(candidates):
    print("\n=== Correctness Check ===")
    results = []
    for name, fn in candidates:
        try:
            ok = quick_correctness_check(fn)
            status = "PASS" if ok else "FAIL"
            print(f"  {name}: {status}")
            if not ok:
                continue
        except Exception as e:
            print(f"  {name}: CRASH ({e})")
            continue
        results.append((name, fn))
    return results


def _benchmark_candidates(results, base_score):
    print(f"\n=== Benchmark ({len(results)} candidates) ===")
    scores = []
    for name, fn in results:
        try:
            score = evaluate_candidate(fn)
            details = evaluate_candidate_detailed(fn)
            geo_ms = 1.0 / score * 1000
            pct = (base_score / score - 1) * 100
            scores.append((score, name, geo_ms, pct, details))
            label = "▲" if score > base_score else "▼" if score < base_score else "="
            print(f"  {name}: {score:.6f} ({geo_ms:.4f}ms) {label} {abs(pct):.2f}%")
            for bn, d in details.items():
                if isinstance(d, dict) and "error" not in d:
                    print(
                        f"      {bn}: median={d['median_ms']:.3f}ms  cv={d['cv']:.4f}"
                    )
        except Exception as e:
            print(f"  {name}: BENCHMARK ERROR ({e})")
    return scores


def _print_final(scores, base_score):
    if not scores:
        return
    scores.sort(reverse=True)
    best_score, best_name, best_ms, best_pct, _ = scores[0]

    print("\n=== Results ===")
    print(f"  Baseline:  {base_score:.6f} ({1.0 / base_score * 1000:.4f}ms)")
    print(
        f"  Champion:  {best_name} ({best_score:.6f}, {best_ms:.4f}ms, {best_pct:+.2f}% vs baseline)"
    )

    print("\n  Rank table:")
    print(f"  {'rank':<5} {'name':<25} {'score':<12} {'ms':<10} {'%':<10}")
    print(f"  {'-' * 5} {'-' * 25} {'-' * 12} {'-' * 10} {'-' * 10}")
    for i, (s, n, ms, pct, _) in enumerate(scores):
        label = "★" if i == 0 else " "
        print(f"  {i + 1:<4} {label} {n:<23} {s:<12.6f} {ms:<10.4f} {pct:<+10.2f}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true", help="Only run correctness check"
    )
    parser.add_argument("--candidate", type=str, help="Run a specific candidate only")
    args = parser.parse_args()

    base_score = _print_baseline()

    candidates = _load_candidates(args.candidate)
    if not candidates:
        print("No candidates to test.")
        return

    results = _check_candidates(candidates)
    if args.quick:
        return

    scores = _benchmark_candidates(results, base_score)
    _print_final(scores, base_score)


if __name__ == "__main__":
    main()
