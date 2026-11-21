#!/usr/bin/env python3
"""
Open a DiskANN disk index without exhausting the kernel's asynchronous-I/O contexts (11-14).

DiskANN's Linux reader calls io_setup(MAX_EVENTS = 1024) once per search thread when the index is
opened; the system-wide limit /proc/sys/fs/aio-max-nr is 65536 here (no root to raise it), i.e. 64
thread contexts for ALL DiskANN processes together. Beyond that the constructor aborts the process
("io_setup() failed with EAGAIN", std::system_error), which killed workload-recall jobs when three
24-thread jobs and two index studies ran at once. open_disk_index waits until enough events are
free for its thread count, under a file lock held until the index is open (the contexts are
allocated in the constructor, so the next waiter sees them in aio-nr).
"""

import os
import time
import fcntl
import logging

logger = logging.getLogger(__name__)
EVENTS_PER_THREAD = 1024
QUEUE_DIR = os.environ.get("RECAL_QUEUE_DIR", "/tmp/mx")
TMP_DIR = os.environ.get("RECAL_TMP", "/tmp")
LOCK = (f"{QUEUE_DIR}/aio.lock" if os.path.isdir(QUEUE_DIR)
        else f"{TMP_DIR}/diskann_aio.lock")


def _read(p):
    with open(p) as f:
        return int(f.read().split()[0])


def open_disk_index(max_wait=None, **kwargs):
    """diskannpy.StaticDiskIndex(**kwargs) once enough AIO events are free (or after max_wait s).
    The lock is held only to check and open, never while sleeping (11-14: a waiter that holds it
    and is then suspended by a quiet timing batch would block every timing run)."""
    import diskannpy
    need = EVENTS_PER_THREAD * int(kwargs.get("num_threads", 1))
    t0 = last = time.time()
    while True:
        with open(LOCK, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                try:
                    used, cap = _read("/proc/sys/fs/aio-nr"), _read("/proc/sys/fs/aio-max-nr")
                except OSError:
                    used, cap = 0, need
                if used + need <= cap or need > cap or (max_wait is not None and time.time() - t0 > max_wait):
                    return diskannpy.StaticDiskIndex(**kwargs)
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)
        if time.time() - last > 600:
            logger.info(f"waiting for AIO contexts: need {need}, {cap - used} free "
                        f"({(time.time() - t0) / 60:.0f} min)")
            last = time.time()
        time.sleep(20)
