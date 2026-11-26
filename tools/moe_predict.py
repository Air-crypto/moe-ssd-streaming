"""How predictable is DeepSeek's expert routing? Analysis of a llama.cpp MoE cache trace (LLAMA_MOE_CACHE_TRACE).

    LLAMA_CPP=/path/to/llama.cpp python3 tools/moe_predict.py <trace.bin> [<trace.bin> ...]

Record a trace with LLAMA_MOE_CACHE_TRACE=trace.bin llama-server ... (decode some tokens, then stop it).

For each decode token (single-token records) and layer, it compares the experts actually selected with:
  - cache:     a simulated LRU of the given size (what the GPU expert cache already catches)
  - prev-tok:  the experts the previous token used at the same layer
  - cooc:      experts co-selected with the current layer's experts in the layer below (learned on the first half,
               scored on the second half): the "predict the next layer" prefetch
  - union:     prev-tok + cooc
Reported as recall (share of selected experts predicted) at a prefetch budget of k experts per layer, and, more
to the point, the share of cache *misses* the predictor would have caught in advance.
"""
import collections
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA_CPP = os.environ.get("LLAMA_CPP", os.path.join(HERE, "..", "llama.cpp"))
spec = importlib.util.spec_from_file_location("tune", os.path.join(LLAMA_CPP, "scripts", "moe-cache-tune.py"))
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)


def decode_steps(records, n_used):
    """[(layer -> tuple(experts))] for each single-token step, in order"""
    steps, cur, first = [], {}, min(il for il, _, _ in records)
    for il, nu, ids in records:
        if len(ids) != nu:          # prompt batches: skip, but they end a decode run
            if cur:
                steps.append(cur)
            cur = {}
            continue
        if il == first and cur:
            steps.append(cur)
            cur = {}
        cur[il] = tuple(ids)
    if cur:
        steps.append(cur)
    return [s for s in steps if len(s) > 1]


def main(paths):
    trace = tune.concat_traces(paths)
    n_layer, n_used, groups, records = trace
    steps = decode_steps(records, n_used)
    layers = sorted({il for s in steps for il in s})
    n_expert = groups[0].n_expert
    print(f"{len(steps)} decode tokens, {len(layers)} MoE layers, {n_expert} experts/layer, {n_used} used per token")

    # simulated cache: per-layer LRU with the same share of slots as the real cache at a few sizes
    def lru_misses(slots_per_layer):
        lru = {il: collections.OrderedDict() for il in layers}
        miss = []   # per step: {il: set(missed experts)}
        for s in steps:
            m = {}
            for il, ex in s.items():
                c = lru[il]
                m[il] = {e for e in ex if e not in c}
                for e in ex:
                    c[e] = True
                    c.move_to_end(e)
                while len(c) > slots_per_layer:
                    c.popitem(last=False)
            miss.append(m)
        return miss

    half = len(steps) // 2
    # co-occurrence between layer il and the next layer, learned on the first half
    nxt = {il: layers[i + 1] for i, il in enumerate(layers[:-1])}
    cooc = {il: collections.defaultdict(collections.Counter) for il in nxt}
    for s in steps[:half]:
        for il, j in nxt.items():
            if il in s and j in s:
                for a in s[il]:
                    cooc[il][a].update(s[j])

    def predict(t, j, k, kind):
        s, prev = steps[t], steps[t - 1] if t > 0 else {}
        out = []
        if kind in ("prev", "union") and j in prev:
            out += list(prev[j])
        if kind in ("cooc", "union"):
            il = next((a for a, b in nxt.items() if b == j), None)
            if il is not None and il in s:
                votes = collections.Counter()
                for a in s[il]:
                    votes.update(cooc[il][a])
                out += [e for e, _ in votes.most_common(k)]
        seen, res = set(), []
        for e in out:
            if e not in seen:
                seen.add(e)
                res.append(e)
        return set(res[:k])

    for frac, label in [(0.12, "~12% of experts cached"), (0.30, "~30% of experts cached")]:
        slots = int(n_expert * frac)
        miss = lru_misses(slots)
        tot_sel = sum(len(s[j]) for s in steps[half:] for j in s)
        tot_miss = sum(len(miss[t][j]) for t in range(half, len(steps)) for j in steps[t])
        print(f"\n{label}: simulated hit rate {1 - tot_miss / tot_sel:.1%} on the scored half")
        print(f"  {'predictor':10s} {'k':>3s}  recall  misses caught  extra reads per useful one")
        for kind in ("prev", "cooc", "union"):
            for k in (6, 12, 24):
                hit = caught = wasted = 0
                for t in range(half, len(steps)):
                    for j, ex in steps[t].items():
                        p = predict(t, j, k, kind)
                        hit += len(p & set(ex))
                        caught += len(p & miss[t][j])
                        wasted += len(p - set(ex))
                print(f"  {kind:10s} {k:3d}  {hit / tot_sel:6.1%}  {caught / max(tot_miss, 1):12.1%}  "
                      f"{wasted / max(caught, 1):8.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
