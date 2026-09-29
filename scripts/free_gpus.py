"""Print the IDs of the least-used GPUs on this node

Orders GPUs by memory in use, then utilization, each summed over several
samples since both fluctuate. Chosen when a job starts.

Standard library only, so that it runs in any image and on the AP.
"""

import subprocess
import sys
import time
from collections import defaultdict

NUM_SAMPLES = 10
INTERVAL = 0.5  # seconds between samples


def free_gpus(num_gpus: int) -> list[str]:
    usage = defaultdict(lambda: [0.0, 0.0])
    for i in range(NUM_SAMPLES):
        if i > 0:
            time.sleep(INTERVAL)
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for line in output.strip().splitlines():
            index, memory, utilization = line.split(", ")
            usage[index][0] += float(memory)
            usage[index][1] += float(utilization)

    if len(usage) < num_gpus:
        raise RuntimeError(
            f"Asked for {num_gpus} GPUs, but found {len(usage)}"
        )
    return sorted(usage, key=usage.get)[:num_gpus]


if __name__ == "__main__":
    print(",".join(free_gpus(int(sys.argv[1]))))  # noqa: T201
