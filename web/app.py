import asyncio
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from .labs import ROOT, LabRegistry, default_params, public_view, without_commentary
from .offscreen import RenderError, RenderPool
from .streaming import STILL_COMPRESSION, StreamSession, bounded_size

class RevalidatedStatic(StaticFiles):

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX = STATIC_DIR / "index.html"
SIZE_PARAMS = ("width", "height")


async def prewarm(registry, renderers):
    for manifest in registry:
        for demo in manifest.get("demos", ()):
            name = f"{manifest['id']}/{demo['id']}"
            started = time.monotonic()
            try:
                await renderers.acquire((manifest["id"], demo["id"]), ROOT,
                                        manifest["srcDir"], demo["scene"], demo.get("options"))
            except Exception as exc:
                print(f"[hub] {name} did not warm up: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
            else:
                print(f"[hub] warmed {name} in {time.monotonic() - started:.1f}s")


@asynccontextmanager
async def lifespan(app):
    app.state.registry = LabRegistry.discover()
    app.state.renderers = RenderPool()
    print(f"[hub] labs found: {len(app.state.registry)}")

    warming = asyncio.create_task(prewarm(app.state.registry, app.state.renderers))
    yield
    warming.cancel()
    app.state.renderers.shutdown()


app = FastAPI(title="KPI lab-hub", lifespan=lifespan)


def registry():
    return app.state.registry


def find_lab(lab_id):
    manifest = registry().lab(lab_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"unknown lab '{lab_id}'")
    return manifest


def find_demo(manifest, demo_id):
    demo = registry().demo(manifest, demo_id)
    if demo is None:
        raise HTTPException(status_code=404, detail=f"unknown demo '{demo_id}'")
    return demo


async def renderer_for(lab_id, demo_id):
    manifest = find_lab(lab_id)
    demo = find_demo(manifest, demo_id)
    try:
        worker = await app.state.renderers.acquire(
            (lab_id, demo_id), ROOT, manifest["srcDir"], demo["scene"], demo.get("options"))
    except RenderError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return worker, demo


@app.get("/api/labs")
async def list_labs():
    return {"labs": registry().public()}


@app.get("/api/labs/{lab_id}")
async def read_lab(lab_id: str):
    return public_view(find_lab(lab_id))


@app.get("/api/labs/{lab_id}/source")
async def read_source(lab_id: str):
    manifest = find_lab(lab_id)
    code = manifest.get("code")
    if not code:
        raise HTTPException(status_code=404, detail="this lab declares no listing")

    src_dir = Path(manifest["srcDir"]).resolve()
    listing = (src_dir / code["file"]).resolve()
    if listing.parent != src_dir or not listing.is_file():
        raise HTTPException(status_code=404, detail="listing file is missing")

    return {
        "label": code.get("label", code["file"]),
        "library": code.get("library", ""),
        "text": without_commentary(listing.read_text(encoding="utf-8")),
    }


@app.get("/api/labs/{lab_id}/demos/{demo_id}/frame.png")
async def render_frame(lab_id: str, demo_id: str, request: Request):
    worker, demo = await renderer_for(lab_id, demo_id)

    query = dict(request.query_params)
    width = bounded_size(query.pop("width", None))
    height = bounded_size(query.pop("height", None))

    params = default_params(demo)
    for name, value in query.items():
        try:
            params[name] = float(value)
        except ValueError:
            continue

    try:
        frame = await asyncio.to_thread(worker.render, width, height, params, STILL_COMPRESSION)
    except RenderError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return Response(content=frame.png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.websocket("/api/labs/{lab_id}/demos/{demo_id}/stream")
async def stream_frames(websocket: WebSocket, lab_id: str, demo_id: str):
    await websocket.accept()
    try:
        worker, demo = await renderer_for(lab_id, demo_id)
    except HTTPException as exc:
        await websocket.send_json({"type": "error", "detail": exc.detail})
        await websocket.close()
        return

    session = StreamSession(worker, demo)
    await websocket.send_json({"type": "hello", "demo": demo, "metadata": worker.metadata})

    pump = asyncio.create_task(session.pump(websocket))
    try:
        while True:
            session.apply(await websocket.receive_json())
    except WebSocketDisconnect:
        pass
    finally:
        pump.cancel()


@app.get("/{course}/{lab}")
async def lab_page(course: str, lab: str):
    slug = f"{course}/{lab}"
    if not registry().has_slug(slug):
        raise HTTPException(status_code=404, detail=f"unknown page '{slug}'")
    return FileResponse(INDEX, headers={"Cache-Control": "no-cache"})


@app.get("/{course}/{lab}/")
async def lab_page_slash(course: str, lab: str):
    return RedirectResponse(f"/{course}/{lab}", status_code=308)


app.mount("/", RevalidatedStatic(directory=STATIC_DIR, html=True), name="static")
