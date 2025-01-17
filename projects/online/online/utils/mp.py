import time
from multiprocessing import Process, Queue
from queue import Empty
from typing import Callable


def process_queue(queue: Queue, f: Callable, sleep: float = 1e-3):
    while True:
        try:
            args = queue.get()
            if args[0] is None:
                continue
            f(*args)
        except Empty:
            time.sleep(sleep)


def initialize_queue_processor(f: Callable, sleep: float) -> Queue:
    queue = Queue()
    queue.put((None,))
    process = Process(target=process_queue, args=(queue, f, sleep))
    process.start()
    return queue
