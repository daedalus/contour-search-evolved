#!/usr/bin/env python3
"""
Evolutionary search loop for contour_search.

Usage:
  python alphaevolve/evolve.py                           # dry run (programmatic mutations only)
  python alphaevolve/evolve.py --llm-cmd "python3 llm_mutate.py"  # real LLM
"""

import sys
import os
import re
import time
import json
import random
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from alphaevolve.evaluator import evaluate_candidate, extract_algorithm, load_candidate_from_file
from search.algorithms import contour_search as baseline_fn

BASELINE_CODE = Path('search/algorithms.py').read_text()
# Extract just the contour_search function
_start_marker = "def contour_search("
_end_marker = "\n\n\ndef "
_start_idx = BASELINE_CODE.find(_start_marker)
_end_idx = BASELINE_CODE.find(_end_marker, _start_idx + 1)
if _start_idx >= 0 and _end_idx >= 0:
    BASELINE_FN = BASELINE_CODE[_start_idx:_end_idx]
else:
    BASELINE_FN = ""
# Add imports needed to run standalone
BASELINE_IMPORT = """from __future__ import annotations
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set
from search.graph import Edge, Graph
"""


@dataclass
class Program:
    code: str
    score: float
    generation: int
    lineage: str = ""


class ProgramDatabase:
    def __init__(self, initial: Program):
        self.programs: List[Program] = [initial]

    def sample_for_mutation(self, k: int = 3) -> List[Program]:
        top = sorted(self.programs, key=lambda p: p.score, reverse=True)
        cutoff = max(1, len(top) // 2)
        pool = top[:cutoff] if random.random() < 0.8 else top
        return random.sample(pool, min(k, len(pool)))

    def add(self, program: Program) -> None:
        self.programs.append(program)
        if len(self.programs) > 200:
            self.programs = sorted(
                self.programs, key=lambda p: p.score, reverse=True
            )[:100]

    @property
    def best(self) -> Program:
        return max(self.programs, key=lambda p: p.score)

    @property
    def best_score(self) -> float:
        return self.best.score

    @property
    def diversity(self) -> float:
        if len(self.programs) < 2:
            return 0.0
        codes = [p.code for p in self.programs]
        diffs = 0.0
        pairs = 0
        for i in range(min(len(codes), 20)):
            for j in range(i + 1, min(len(codes), 20)):
                a, b = codes[i], codes[j]
                max_len = max(len(a), len(b))
                if max_len == 0:
                    continue
                diff = sum(1 for c1, c2 in zip(a, b) if c1 != c2)
                diff += abs(len(a) - len(b))
                diffs += diff / max_len
                pairs += 1
        return diffs / pairs if pairs > 0 else 0.0


def build_mutation_prompt(
    parents: List[Program],
    best_score: float,
    baseline_score: float,
    instruction: str = "",
) -> str:
    parent_strs = []
    for i, p in enumerate(parents):
        label = chr(65 + i)
        parent_strs.append(f"--- Algorithm {label} (score: {p.score:.4f}) ---\n{p.code}")

    return textwrap.dedent(f"""\
    You are an expert algorithm designer optimizing for SPEED.
    Your task is to improve the `contour_search` function to make it FASTER.

    The algorithm: contour-based A* search. It groups nodes into buckets by
    quantized f = g + heuristic value, then expands all nodes in the cheapest
    bucket before moving to the next.

    Current best score: {best_score:.4f}
    Baseline score: {baseline_score:.4f}
    Higher score = faster runtime across chain, star, grid, and dense graphs.

    {chr(10).join(parent_strs)}

    {instruction}

    Propose a new version that might be faster. Output ONLY valid Python code.
    The function must have this exact signature:
    def contour_search(
        graph: Graph,
        start: str,
        goal: str,
        heuristic: Callable[[str, str], float] = lambda a, b: 0.0,
        precision: int = 3,
    ) -> Optional[List[str]]:
    """)


def extract_code(text: str) -> Optional[str]:
    match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    if stripped.startswith('def contour_search'):
        return stripped
    return None


# ---------------------------------------------------------------------------
# Programmatic mutations (no LLM needed)
# ---------------------------------------------------------------------------

def mutate_min_optimization(code: str) -> str:
    """Use local variable bindings to avoid repeated lookups."""
    return code.replace(
        "def contour_search(",
        "def contour_search("
    )
    return code


def mutate_local_refs(code: str) -> str:
    """Speed up by caching globals as locals."""
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('graph.neighbors'):
            lines[i] = line.replace('graph.neighbors', 'neighbors')
    
    for i, line in enumerate(lines):
        if line.strip().startswith('visited:'):
            indent = line[:len(line) - len(line.lstrip())]
            lines.insert(i + 1, f'{indent}neighbors = graph.neighbors')
            break
    return '\n'.join(lines)


def mutate_use_min_iter(code: str) -> str:
    """Replace min(buckets.keys()) with iterative min tracking."""
    if 'current_min = None' in code:
        return code
    lines = code.split('\n')
    new_lines = []
    patched = False
    for line in lines:
        new_lines.append(line)
        if 'buckets.setdefault(f_start, []).append' in line and not patched:
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(f'{indent}current_min = f_start')
            patched = True
    return '\n'.join(new_lines).replace(
        'min_f = min(buckets.keys())',
        'min_f = current_min\n        if min_f not in buckets:\n            min_f = min(buckets.keys())\n        current_min = min_f'
    )


def mutate_deque_buckets(code: str) -> str:
    """Use deque.pop for bucket iteration instead of list iteration."""
    return code  # placeholder


PROGRAMMATIC_MUTATIONS = [
    mutate_local_refs,
    mutate_min_optimization,
    mutate_use_min_iter,
]


def call_llm(prompt: str, llm_cmd: Optional[str] = None) -> str:
    """Call LLM. If no command given, use a programmatic mutation instead."""
    if llm_cmd:
        result = subprocess.run(
            llm_cmd.split() + [prompt],
            capture_output=True, text=True, timeout=60
        )
        return result.stdout
    # Fallback: random programmatic mutation
    mutation = random.choice(PROGRAMMATIC_MUTATIONS)
    return mutation(prompt)


def run_generation(
    db: ProgramDatabase,
    generation: int,
    llm_cmd: Optional[str] = None,
) -> Program:
    """Run one generation of evolution."""
    parents = db.sample_for_mutation()
    instruction = ""
    
    if generation > 5 and db.best_score < 0.01:
        instruction = "Think radically differently. Try a completely new approach."
    elif db.diversity < 0.3:
        instruction = "Try a different approach than existing solutions."

    prompt = build_mutation_prompt(
        parents, db.best_score, db.best_score, instruction
    )
    
    response = call_llm(prompt, llm_cmd)
    code = extract_code(response)
    
    if not code:
        return None
    
    full_code = BASELINE_IMPORT + "\n\n" + code
    
    try:
        candidate_fn = extract_algorithm(full_code)
    except Exception as e:
        return None
    
    try:
        score = evaluate_candidate(candidate_fn)
    except Exception as e:
        return None
    
    lineage = ",".join(chr(65 + parents.index(p)) if p in parents else "?" for p in parents)
    prog = Program(code=code, score=score, generation=generation, lineage=lineage)
    
    if score >= db.best_score * 0.95:
        db.add(prog)
    
    return prog


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--llm-cmd", type=str, default=None,
                        help="Command to call LLM. Receives prompt on stdin, returns code on stdout.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Baseline
    baseline_full = BASELINE_IMPORT + "\n\n" + BASELINE_FN
    baseline_score = evaluate_candidate(baseline_fn)
    db = ProgramDatabase(Program(
        code=BASELINE_FN,
        score=baseline_score,
        generation=0,
        lineage="baseline",
    ))
    
    print(f"{'gen':>4} {'score':>10} {'vs_baseline':>12} {'diversity':>10}  notes")
    print(f"{'─'*40}")
    print(f"{0:>4} {baseline_score:>10.4f} {'1.000x':>12} {db.diversity:>10.3f}  baseline")
    
    best_so_far = db.best
    
    for gen in range(1, args.generations + 1):
        prog = run_generation(db, gen, args.llm_cmd)
        
        if prog is None:
            continue
        
        ratio = prog.score / baseline_score
        flag = ""
        if prog.score > best_so_far.score:
            best_so_far = prog
            flag = " ★ NEW BEST"
        elif prog.score >= baseline_score * 1.5:
            flag = " (good)"
        
        print(f"{gen:>4} {prog.score:>10.4f} {ratio:>11.3f}x {db.diversity:>10.3f} {flag}")
    
    print(f"\n{'='*50}")
    print(f"Best score: {best_so_far.score:.4f} vs baseline {baseline_score:.4f}")
    print(f"Speedup: {baseline_score / best_so_far.score:.2f}x" if best_so_far.score > 0 else "N/A")
    
    out_path = Path("alphaevolve/best_found.py")
    out_path.write_text(best_so_far.code)
    print(f"Best code written to {out_path}")


if __name__ == "__main__":
    main()
