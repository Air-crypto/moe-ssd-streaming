#!/usr/bin/env python3
"""Random-read throughput of the drive holding a model, at several read sizes and queue depths.

    python3 tools/ssdbench.py model-00001-of-00008.gguf [more files...]

Uses F_NOCACHE (macOS) so it measures the drive, not the page cache. Compare the 48 KiB x QD1 line (roughly
what page faults on an mmap'd model give you) with the multi-MiB parallel lines (what the patch uses).
"""
import fcntl
import os
import random
import sys
import threading
import time

F_NOCACHE = 48  # macOS


def open_all(paths):
    out = []
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            fcntl.fcntl(fd, F_NOCACHE, 1)
        except OSError:
            pass  # not macOS: results may include page-cache hits
        out.append((fd, os.fstat(fd).st_size))
    return out


def run(fds, size, qd, total=4 << 30):
    n = max(qd, total // size)
    done = [0]
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                if done[0] >= n:
                    return
                done[0] += 1
            fd, fsz = random.choice(fds)
            off = random.randrange(0, fsz - size) // 16384 * 16384
            os.pread(fd, size, off)

    t = time.time()
    th = [threading.Thread(target=worker) for _ in range(qd)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    dt = time.time() - t
    print(f"read {size >> 10:6d} KiB x QD{qd:<3d}: {n * size / dt / 1e9:5.2f} GB/s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    fds = open_all(sys.argv[1:])
    for size, qd in [(48 << 10, 1), (48 << 10, 16), (1 << 20, 1), (1 << 20, 16),
                     (6 << 20, 1), (6 << 20, 4), (6 << 20, 16)]:
        run(fds, size, qd)
