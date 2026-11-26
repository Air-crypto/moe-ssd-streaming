#!/bin/bash
# Decode benchmark against a running llama-server: 3 realistic prompts, thinking off, prints prompt and
# generation tok/s from llama.cpp's own timings.
#   tools/bench.sh [http://127.0.0.1:8080] [max_tokens]
URL=${1:-http://127.0.0.1:8080}
N=${2:-256}
for p in "Write a Python function that merges overlapping intervals, with a docstring and two examples." \
         "Explain how a CPU cache hierarchy works to a junior engineer, in about 200 words." \
         "Give me a 5-day itinerary for Kyoto with one sentence per day."; do
  body=$(python3 -c 'import json,sys; print(json.dumps({"messages":[{"role":"user","content":sys.argv[1]}],
    "max_tokens":int(sys.argv[2]),"reasoning_effort":"none","temperature":1.0,"top_p":1.0}))' "$p" "$N")
  curl -s -m 3600 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" | python3 -c '
import json,sys; r=json.load(sys.stdin); t=r["timings"]
print("prompt %4d tok @ %6.1f t/s | gen %3d tok @ %5.2f t/s" % (t["prompt_n"], t["prompt_per_second"], t["predicted_n"], t["predicted_per_second"]))'
done
