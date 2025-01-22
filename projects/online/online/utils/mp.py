from multiprocessing import Process, Queue
from typing import Callable


def process_queue(queue: Queue, f: Callable):
    while True:
        args = queue.get()
        f(*args)


def initialize_queue_processor(f: Callable) -> Queue:
    queue = Queue()
    process = Process(target=process_queue, args=(queue, f))
    process.start()
    return queue
