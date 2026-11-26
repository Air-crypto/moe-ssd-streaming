# moe-ssd-streaming

Running a 650 GB mixture-of-experts model on a 128 GB Mac by streaming its experts from the internal SSD,
with a small set of llama.cpp changes that make that about **10× faster** than stock llama.cpp.

Test model: DeepSeek-V4.1-Flash (40 MoE layers, 384 routed experts per layer, 6 active per token), converted
losslessly to GGUF. That gives 269 GiB of MXFP4 experts, 367 GiB of Engram lookup tables (read lazily) and about
15 GiB of everything else. Test machine: Apple M5 Max, 128 GB unified memory, internal SSD.

## Results

Generation speed at a single stream, thinking off, 256-token answers to three realistic prompts. Prefill is a
1,838-token prompt. The GPU expert cache is 70 GiB (`--moe-cache-mib 71680`) unless noted. Run it at 60–64 GiB
with DSpark and the mmproj loaded, to keep about 10 GB of RAM free.

| build / setting | decode tok/s | prefill tok/s (1.8k prompt) |
|---|---:|---:|
| stock llama.cpp, `--cpu-moe --moe-cache-mib` | 0.8 – 1.2 | 2.5 (`-ub 512`) |
| + parallel `pread` for cache misses | 6.3 – 7.6 | 14 (`-ub 8`, cache path) |
| + reads straight into the unified-memory cache slots | 8.9 – 9.9 | — |
| + DSpark speculative decoding, 2 draft tokens | 9.4 – 10.9 | — |
| + parallel `pread` for large-batch (prefill) expert copies | same | 27.4 (`-ub 512`), **69.1** (`-ub 2048`) |
| everything + vision projector (`--mmproj`) + DSpark, 64 GiB cache | **8.6 – 11.8** | 65 |

That's up to **10× faster decode and 28× faster prefill** than stock, on the same hardware and settings.

Other results:
- Image understanding still works through the mmproj: 30 tok/s on the image prompt.
- With DSpark at 3 draft tokens instead of 2, code got faster but prose got slower. Acceptance fell to 37–40%,
  and each verification step touches more distinct experts.

## Why stock llama.cpp is slow here

With `--cpu-moe` the experts stay in the mmap'd GGUF and the GPU expert cache (`--moe-cache-mib`) uploads
misses from it. Each upload is a `memcpy` from the mapping, so the data arrives through page faults: one
16–48 KiB read at a time. The drive answers that at about 0.65 GB/s, but it can do far better
(`tools/ssdbench.py`, on the GGUF itself, page cache bypassed):

| read size × parallel reads | GB/s |
|---|---:|
| 48 KiB × 1 (≈ page faults) | 0.65 |
| 48 KiB × 16 | 7.2 |
| 1 MiB × 16 | 16.8 |
| 6 MiB × 4 | 17.4 |

An expert matrix here is about 6 MiB, so a layer's misses are a handful of large, independent reads.

## What the patch does

`patches/0001-moe-ssd-streaming.patch`, against llama.cpp `e8906e6` (2026-11-25):

1. **mmap registry** (`llama-mmap.cpp/h`). Live file mappings are recorded with a dup'd fd. Code holding a
   weight pointer can then look up the file and offset (`llama_mmap_file_range`) and read those bytes with
   `pread` (`llama_mmap_read_ranges`, a fixed pool of readers, 1 MiB pieces).
2. **Parallel reads of cache misses** (`llama-moe-cache.cpp`). Once a layer's selected experts are known, all
   its misses are read at once by 16 threads. When the cache banks are CPU-visible device memory (Metal shared
   buffers on unified memory), the reads land directly in the cache slots, with no staging copy and no upload
   call. The backend has already been synchronized at that point, so nothing is reading those slots.
3. **Parallel reads for large batches** (`llama-context.cpp`, `sched_copy_experts`). Prefill ubatches copy
   every selected expert that isn't cached to the device. Those copies go through the same reader.
4. **Expert prediction and prefetch** (experimental, off by default). `LLAMA_MOE_PREDICT_K=k` adds one small
   node per MoE layer: the next layer's router applied to this layer's expert input, top-k. With
   `LLAMA_MOE_PREFETCH=1`, the predicted misses of the next layer are read in the background while the current
   layer computes. See *Negative result* below.
5. Statistics on stderr every 2,000 cache lookups: read GB/s, hit rate, time waiting for the device vs reading,
   and prediction recall.

| env var | default | |
|---|---|---|
| `LLAMA_MOE_PREAD` | `1` | `0` restores stock behaviour |
| `LLAMA_MOE_PREAD_THREADS` | `16` | readers; 16 and 32 measured the same here |
| `LLAMA_MOE_PREAD_CHUNK_KIB` | `1024` | piece size for cache misses; 256–2048 measured the same |
| `LLAMA_MOE_PREAD_DIRECT` | `1` | `0` reads into a staging buffer and uploads from it |
| `LLAMA_MOE_PREDICT_K` | `0` | experts predicted per layer for the next layer |
| `LLAMA_MOE_PREFETCH` | `0` | `1` prefetches the predicted misses |

## Where the time goes now

Measured from the patch's own counters during decode, about 700 tokens:

- **Reading misses from the SSD: about 40%.** Reads run at about 12.4 GB/s, and the cache hit rate is about
  86%. Each layer misses only about 1 expert per token, which is too little work per call to reach the drive's
  17 GB/s.
- **Waiting for the GPU at cache lookups: about 5%.** The GPU is mostly idle.
- **Everything else: the remaining ~55%.** That covers attention, the Engram lookups, graph and scheduler
  overhead, and the CPU parts.

Before item 2 removed it, another 30% went into copying from a staging buffer into the cache.

## Negative result: expert prediction

Applying layer N+1's router to layer N's expert input predicts layer N+1's experts well:

| predicted experts | share of the next layer's selected experts |
|---:|---:|
| 6 | 70.9% |
| 12 | 85.8% |

Routing history alone does far worse (`tools/moe_predict.py` on a trace): the previous token's experts, or
co-occurrence with the layer below, catch only 3–25% of the misses.

Prefetching the predicted misses still made decode **slower**: 7.3–8.1 tok/s with k=6 and 6.5–7.7 with k=8,
against 8.9–9.9 without prefetch. There are two reasons:

- With 86% of selections already cached, the misses are the rare experts, which are the hardest to predict.
  Only 28–41% of the prefetched experts were used.
- The wasted reads compete with the demand reads for the drive, and prefetching only one layer ahead leaves
  little time to hide them.

Prediction would pay off if the hit rate were much lower (a smaller cache), or if prefetch ran several layers
ahead with high-precision filtering.

## Prefill notes

A prefill ubatch selects nearly every expert in every layer, so each ubatch streams most of the 269 GiB. Larger
ubatches mean fewer passes over the experts: 1,838 tokens took 722 s at `-ub 512` on stock llama.cpp, 67 s at
`-ub 512` with the patch, and 27 s at `-ub 2048`. The cost is a bigger compute buffer, so check free memory.

| ubatch | prefill tok/s (patched) |
|---:|---:|
| 8 | 14.0 |
| 32 | 14.6 |
| 512 | 27.4 |
| 2048 | 69.1 |

## Reproduce

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout e8906e6
git apply ../moe-ssd-streaming/patches/0001-moe-ssd-streaming.patch
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-server

# MoE weights stay in the mmap; no CPU repack (it would copy all experts into RAM)
build/bin/llama-server -m model-00001-of-00008.gguf -ngl 999 --cpu-moe --no-repack \
    --moe-cache-mib 61440 -c 32768 -np 1 -fa on --jinja -ub 2048 -b 2048 \
    --spec-type draft-dspark -md dspark.gguf -ngld 999 --spec-draft-n-max 2 \
    --mmproj mmproj.gguf --chat-template-file chat_template.jinja
tools/bench.sh http://127.0.0.1:8080
```

Gotchas found on the way:
- **`--no-repack` is required.** Without it, CPU repacking copies all 269 GiB of experts into anonymous memory
  and the machine swaps until the process is killed.
- **DeepSeek-V4.1 chat template:** if your GGUF has no chat template (the source repo lacked
  `chat_template.jinja`), pass the upstream template with `--chat-template-file`. Otherwise llama.cpp falls back
  to ChatML and the model never stops.
- **`tools/watchdog.sh`** kills the server if free RAM drops below 6 GB or swap grows. It's useful while
  finding the largest safe cache size.

## Tools

- `tools/ssdbench.py`: drive throughput by read size and queue depth, on your model files.
- `tools/bench.sh`: decode and prefill tok/s against a running server.
- `tools/moe_predict.py`: routing predictability from a `LLAMA_MOE_CACHE_TRACE` trace. Uses llama.cpp's
  `scripts/moe-cache-tune.py` parser; set `LLAMA_CPP` to your llama.cpp checkout.
- `tools/watchdog.sh`: memory watchdog for experiments.

License: MIT for the tools; the patch follows llama.cpp's MIT license.
