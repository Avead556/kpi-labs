import asyncio
import multiprocessing as mp
import queue
import sys
import threading
from collections import namedtuple

START_TIMEOUT = 30.0
RENDER_TIMEOUT = 20.0
SHUTDOWN_TIMEOUT = 5.0


Frame = namedtuple("Frame", "png readout")


class RenderError(RuntimeError):
    pass


def _worker_main(root, src_dir, scene_name, options, requests, responses):
    sys.path[:0] = [root, src_dir]
    try:
        import scenes
        from labkit import glkit

        glkit.create_offscreen_context()
        scene = scenes.create_scene(scene_name, options)
        target = glkit.Offscreen()
        responses.put(("ready", scene.metadata()))
    except Exception as exc:
        responses.put(("fatal", f"{type(exc).__name__}: {exc}"))
        return

    for request in iter(requests.get, None):
        width, height, params, compression = request
        try:
            size = target.bind(width, height)
            scene.draw(size[0], size[1], params)
            readout = scene.readout(params) if hasattr(scene, "readout") else None
            responses.put(("frame", (target.read_png(compression), readout)))
        except Exception as exc:
            responses.put(("error", f"{type(exc).__name__}: {exc}"))


class RenderWorker:

    def __init__(self, root, src_dir, scene_name, options=None):
        self.root = str(root)
        self.src_dir = str(src_dir)
        self.scene_name = scene_name
        self.options = options or {}
        self.metadata = {}
        self._lock = threading.Lock()
        self._process = None
        self._requests = None
        self._responses = None

    def start(self):
        context = mp.get_context("spawn")
        self._requests = context.Queue()
        self._responses = context.Queue()
        self._process = context.Process(
            target=_worker_main,
            args=(self.root, self.src_dir, self.scene_name, self.options,
                  self._requests, self._responses),
            daemon=True,
        )
        self._process.start()

        kind, payload = self._await_reply(START_TIMEOUT, "start")
        if kind != "ready":
            self.stop()
            raise RenderError(f"scene '{self.scene_name}' failed to start: {payload}")
        self.metadata = payload
        return self

    def render(self, width, height, params=None, compression=1):
        with self._lock:
            if self._process is None or not self._process.is_alive():
                raise RenderError(f"render process for '{self.scene_name}' is not running")
            self._requests.put((width, height, params or {}, compression))
            kind, payload = self._await_reply(RENDER_TIMEOUT, "render")
        if kind != "frame":
            raise RenderError(payload)
        return Frame(*payload)

    def _await_reply(self, timeout, action):
        try:
            return self._responses.get(timeout=timeout)
        except queue.Empty:
            if action == "start":
                self.stop()
            raise RenderError(
                f"{action} of '{self.scene_name}' timed out after {timeout:.0f}s")

    def stop(self):
        with self._lock:
            process, self._process = self._process, None
            if process is None:
                return
            if process.is_alive():
                self._requests.put(None)
                process.join(timeout=SHUTDOWN_TIMEOUT)
            if process.is_alive():
                process.terminate()


class RenderPool:

    def __init__(self):
        self._workers = {}
        self._locks = {}
        self._guard = asyncio.Lock()

    async def acquire(self, key, root, src_dir, scene_name, options=None):
        worker = self._workers.get(key)
        if worker is not None:
            return worker

        async with self._guard:
            lock = self._locks.setdefault(key, asyncio.Lock())

        async with lock:
            worker = self._workers.get(key)
            if worker is None:
                worker = RenderWorker(root, src_dir, scene_name, options)
                await asyncio.to_thread(worker.start)
                self._workers[key] = worker
            return worker

    def shutdown(self):
        for worker in self._workers.values():
            worker.stop()
        self._workers.clear()
