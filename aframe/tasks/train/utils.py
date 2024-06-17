import os
import shlex
import subprocess
import sys
from queue import Queue
from threading import Thread
from typing import List


def read_stream(stream, process, q):
    stream = getattr(process, stream)
    try:
        it = iter(stream.readline, b"")
        while True:
            try:
                line = next(it)
            except StopIteration:
                break
            q.put(line.decode())
    finally:
        q.put(None)


def stream_process(process):
    q = Queue()
    args = (process, q)
    streams = ["stdout", "stderr"]
    threads = [Thread(target=read_stream, args=(i,) + args) for i in streams]
    for t in threads:
        t.start()

    for _ in range(2):
        for line in iter(q.get, None):
            sys.stdout.write(line)


def stream_command(command: List[str]):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ
    )
    stream_process(process)

    process.wait()
    if process.returncode:
        raise RuntimeError(
            "Command '{}' failed with return code {} "
            "and stderr:\n{}".format(
                shlex.join(command),
                process.returncode,
                process.stderr.read().decode(),
            )
        ) from None
