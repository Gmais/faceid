import os
import uuid
import asyncio
import json
import subprocess
import threading
import queue
import shutil
import multiprocessing
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).parent.parent
WEBAPP_DIR = Path(__file__).parent
UPLOADS_DIR = WEBAPP_DIR / "uploads"
OUTPUTS_DIR = WEBAPP_DIR / "outputs"
STATIC_DIR = WEBAPP_DIR / "static"
HELPER_SCRIPT = WEBAPP_DIR / "detect_faces_helper.py"

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
MAX_FILE_MB = 500

PYTHON_EXE = str(BASE_DIR / "venv" / "Scripts" / "python.exe")
FACEFUSION_PY = str(BASE_DIR / "facefusion.py")
THREAD_COUNT = max(4, multiprocessing.cpu_count())

# 0 = sem limite. Vídeos (já com trim aplicado) acima disso são recusados com DURATION_EXCEEDED.
MAX_VIDEO_SECONDS = float(os.environ.get("MAX_VIDEO_SECONDS", "0"))

jobs: Dict[str, dict] = {}
job_queue: "queue.Queue[str]" = queue.Queue()
pending_ids: list = []  # job_ids aguardando na fila, em ordem FIFO
pending_lock = threading.Lock()


def _detect_cuda() -> bool:
    try:
        import onnxruntime
        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False


HAS_CUDA = _detect_cuda()


def cleanup_old_files():
    cutoff = datetime.now() - timedelta(hours=2)
    for directory in [UPLOADS_DIR, OUTPUTS_DIR]:
        for f in directory.iterdir():
            try:
                if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                    f.unlink() if f.is_file() else shutil.rmtree(f)
            except Exception:
                pass


async def _periodic_cleanup():
    while True:
        await asyncio.sleep(1800)
        cleanup_old_files()


def _queue_worker():
    """Um único worker: processa um job por vez para não disputar a GPU."""
    while True:
        job_id = job_queue.get()
        with pending_lock:
            if job_id in pending_ids:
                pending_ids.remove(job_id)
        try:
            if job_id in jobs:
                _run_facefusion(job_id)
        finally:
            job_queue.task_done()


def _queue_position(job_id: str):
    with pending_lock:
        return pending_ids.index(job_id) + 1 if job_id in pending_ids else 0


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_queue_worker, daemon=True).start()
    asyncio.create_task(_periodic_cleanup())
    yield


app = FastAPI(title="FaceSwap Studio", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/api/upload/source")
async def upload_source(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT:
        raise HTTPException(400, f"Imagem obrigatória. Formatos: {', '.join(ALLOWED_IMAGE_EXT)}")

    content = await file.read()
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(400, f"Arquivo muito grande (máx {MAX_FILE_MB}MB)")

    file_id = str(uuid.uuid4())
    dest = UPLOADS_DIR / f"source_{file_id}{ext}"
    dest.write_bytes(content)
    return {"file_id": file_id, "filename": file.filename, "path": str(dest)}


@app.post("/api/upload/target")
async def upload_target(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    all_ext = ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT
    if ext not in all_ext:
        raise HTTPException(400, "Formato não suportado")

    content = await file.read()
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(400, f"Arquivo muito grande (máx {MAX_FILE_MB}MB)")

    file_id = str(uuid.uuid4())
    dest = UPLOADS_DIR / f"target_{file_id}{ext}"
    dest.write_bytes(content)
    is_video = ext in ALLOWED_VIDEO_EXT
    return {"file_id": file_id, "filename": file.filename, "path": str(dest), "is_video": is_video}


@app.get("/api/video-info")
async def video_info(path: str):
    """Returns fps, duration, total_frames for a video file."""
    if not Path(path).exists():
        raise HTTPException(400, "Arquivo não encontrado")
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duration = round(total_frames / fps, 2) if fps > 0 else 0
        return {"fps": fps, "total_frames": total_frames, "width": width, "height": height, "duration": duration}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/detect-faces")
async def detect_faces(request: Request):
    """Detects faces in a video frame or image; returns thumbnails as base64."""
    body = await request.json()
    target_path = body.get("target_path")
    frame_number = body.get("frame_number")  # seconds converted to frame externally, or raw frame index

    if not target_path or not Path(target_path).exists():
        raise HTTPException(400, "Arquivo não encontrado")

    cmd = [PYTHON_EXE, str(HELPER_SCRIPT), target_path]
    if frame_number is not None:
        cmd.append(str(int(frame_number)))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(BASE_DIR),
            env=_build_env(),
        )
        data = json.loads(result.stdout)
        return data
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Tempo esgotado na detecção")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/jobs")
async def create_job(request: Request):
    body = await request.json()
    source_path = body.get("source_path")
    target_path = body.get("target_path")
    processors = body.get("processors", ["face_swapper", "face_enhancer"])
    face_selector_mode = body.get("face_selector_mode", "one")
    trim_start_sec = body.get("trim_start_sec")   # float seconds or null
    trim_end_sec = body.get("trim_end_sec")         # float seconds or null
    reference_face_position = body.get("reference_face_position")  # int or null
    reference_frame_number = body.get("reference_frame_number")    # int or null
    speed_mode = body.get("speed_mode", "quality")  # "fast" or "quality"
    # Modo vários rostos: lista de {source_path, reference_face_position, reference_frame_number}
    pairs = body.get("pairs") or []

    if pairs:
        for pair in pairs:
            if pair.get("reference_face_position") is None or pair.get("reference_frame_number") is None:
                raise HTTPException(400, "Cada par precisa de um rosto de referência")
            if not pair.get("source_path") or not Path(pair["source_path"]).exists():
                raise HTTPException(400, "Rosto substituto não encontrado em um dos pares")
        source_path = pairs[0]["source_path"]
    if not source_path or not target_path:
        raise HTTPException(400, "source_path e target_path são obrigatórios")
    if not Path(source_path).exists():
        raise HTTPException(400, "Arquivo fonte não encontrado")
    if not Path(target_path).exists():
        raise HTTPException(400, "Arquivo alvo não encontrado")

    # Convert seconds → frames when trim is requested
    trim_start_frame = None
    trim_end_frame = None
    target_ext = Path(target_path).suffix.lower()
    if target_ext in ALLOWED_VIDEO_EXT and (trim_start_sec is not None or trim_end_sec is not None):
        try:
            import cv2
            cap = cv2.VideoCapture(target_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if trim_start_sec is not None:
                trim_start_frame = max(0, int(float(trim_start_sec) * fps))
            if trim_end_sec is not None:
                trim_end_frame = min(total_frames, int(float(trim_end_sec) * fps))
        except Exception:
            pass

    if MAX_VIDEO_SECONDS > 0 and target_ext in ALLOWED_VIDEO_EXT:
        try:
            import cv2
            cap = cv2.VideoCapture(target_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            start = trim_start_frame or 0
            end = trim_end_frame if trim_end_frame is not None else total_frames
            if (end - start) / fps > MAX_VIDEO_SECONDS:
                raise HTTPException(400, {"code": "DURATION_EXCEEDED", "max_seconds": MAX_VIDEO_SECONDS})
        except HTTPException:
            raise
        except Exception:
            pass

    job_id = str(uuid.uuid4())
    out_ext = ".mp4" if target_ext in ALLOWED_VIDEO_EXT else target_ext
    output_path = str(OUTPUTS_DIR / f"output_{job_id}{out_ext}")

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "source_path": source_path,
        "target_path": target_path,
        "output_path": output_path,
        "processors": processors,
        "face_selector_mode": face_selector_mode,
        "trim_start_frame": trim_start_frame,
        "trim_end_frame": trim_end_frame,
        "reference_face_position": reference_face_position,
        "reference_frame_number": reference_frame_number,
        "speed_mode": speed_mode,
        "pairs": pairs,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "logs": [],
    }

    with pending_lock:
        pending_ids.append(job_id)
    job_queue.put(job_id)
    return {"job_id": job_id, "queue_position": _queue_position(job_id)}


def _build_env():
    env = os.environ.copy()
    try:
        import winreg
        paths = set(env.get("PATH", "").split(os.pathsep))
        for hive, subkey in [
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            (winreg.HKEY_CURRENT_USER, r"Environment"),
        ]:
            try:
                with winreg.OpenKey(hive, subkey) as k:
                    val, _ = winreg.QueryValueEx(k, "PATH")
                    for p in val.split(os.pathsep):
                        paths.add(os.path.expandvars(p))
            except Exception:
                pass
        env["PATH"] = os.pathsep.join(filter(None, paths))
    except ImportError:
        pass
    return env


def _build_cmd(job: dict, source: str, target: str, output: str, ref_pos, ref_frame, trim: bool) -> list:
    cmd = [
        PYTHON_EXE, FACEFUSION_PY, "headless-run",
        "--source-paths", source,
        "--target-path", target,
        "--output-path", output,
        "--processors", *job["processors"],
        "--log-level", "info",
        "--execution-thread-count", str(THREAD_COUNT),
    ]
    if trim:
        if job.get("trim_start_frame") is not None:
            cmd += ["--trim-frame-start", str(job["trim_start_frame"])]
        if job.get("trim_end_frame") is not None:
            cmd += ["--trim-frame-end", str(job["trim_end_frame"])]

    # Face selection — reference overrides the dropdown mode
    if ref_pos is not None and ref_frame is not None:
        cmd += [
            "--face-selector-mode", "reference",
            "--reference-face-position", str(ref_pos),
            "--reference-frame-number", str(ref_frame),
        ]
    else:
        cmd += ["--face-selector-mode", job.get("face_selector_mode", "one")]

    cmd += ["--output-video-preset", "ultrafast" if job.get("speed_mode") == "fast" else "veryfast"]
    if HAS_CUDA:
        cmd += ["--execution-providers", "cuda"]
    return cmd


def _run_pass(job: dict, cmd: list, base: int, span: int) -> bool:
    """Roda um passe do FaceFusion e mapeia o progresso para [base, base+span]."""
    n_steps = max(1, len(job.get("processors", ["face_swapper"])))
    slice_size = max(1, span // n_steps)
    job["_step"] = 0
    job["_prev_raw_pct"] = 0

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(BASE_DIR),
        encoding="utf-8",
        errors="replace",
        env=_build_env(),
    )
    for line in (proc.stdout or []):
        line = line.strip()
        if not line:
            continue
        job["logs"].append(line)
        if len(job["logs"]) > 200:
            job["logs"] = job["logs"][-200:]

        if "%" in line and "|" in line:
            try:
                raw_pct = int(float(line.split("%")[0].split()[-1]))
                # Detect new processor step: big drop in % means a new step started
                if raw_pct < job["_prev_raw_pct"] - 30:
                    job["_step"] += 1
                job["_prev_raw_pct"] = raw_pct
                step = min(job["_step"], n_steps - 1)
                overall = base + step * slice_size + (raw_pct * slice_size) // 100
                job["progress"] = min(94, overall)
            except Exception:
                pass
    proc.wait()
    return proc.returncode == 0


def _run_facefusion(job_id: str):
    job = jobs[job_id]
    job["status"] = "processing"
    job["progress"] = 5

    # Cada par (rosto do vídeo -> rosto substituto) vira um passe encadeado:
    # a saída de um passe é o alvo do próximo.
    pairs = job.get("pairs") or [{
        "source_path": job["source_path"],
        "reference_face_position": job.get("reference_face_position"),
        "reference_frame_number": job.get("reference_frame_number"),
    }]
    ext = Path(job["output_path"]).suffix
    trim_offset = job.get("trim_start_frame") or 0
    span = 90 // len(pairs)
    temps = []

    try:
        target = job["target_path"]
        for i, pair in enumerate(pairs):
            last = i == len(pairs) - 1
            output = job["output_path"] if last else str(OUTPUTS_DIR / f"tmp_{job_id}_{i}{ext}")
            if not last:
                temps.append(output)
            ref_frame = pair.get("reference_frame_number")
            if i > 0 and ref_frame is not None:
                ref_frame = max(0, ref_frame - trim_offset)  # alvo já está cortado
            if len(pairs) > 1:
                job["logs"].append(f"== Passe {i + 1}/{len(pairs)} ==")
            cmd = _build_cmd(job, pair["source_path"], target, output,
                             pair.get("reference_face_position"), ref_frame, trim=(i == 0))
            if not _run_pass(job, cmd, 5 + i * span, span):
                job["status"] = "failed"
                job["error"] = f"Processamento falhou no passe {i + 1}. Verifique os logs."
                return
            target = output

        if Path(job["output_path"]).exists():
            job["status"] = "completed"
            job["progress"] = 100
        else:
            job["status"] = "failed"
            job["error"] = "Processamento falhou. Verifique os logs."
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
    finally:
        for t in temps:
            try:
                Path(t).unlink(missing_ok=True)
            except Exception:
                pass


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job não encontrado")
    j = jobs[job_id].copy()
    j["logs"] = j["logs"][-30:]
    j["queue_position"] = _queue_position(job_id)
    return j


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job não encontrado")

    async def stream():
        last = 0
        while True:
            if job_id not in jobs:
                break
            j = jobs[job_id]
            new_logs = j["logs"][last:]
            last = len(j["logs"])
            data = json.dumps({
                "status": j["status"],
                "progress": j["progress"],
                "logs": new_logs,
                "error": j["error"],
                "queue_position": _queue_position(job_id),
            })
            yield f"data: {data}\n\n"
            if j["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.6)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/result")
async def download_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job não encontrado")
    j = jobs[job_id]
    if j["status"] != "completed":
        raise HTTPException(400, f"Job está {j['status']}, não concluído")
    out = Path(j["output_path"])
    if not out.exists():
        raise HTTPException(404, "Arquivo de saída não encontrado")
    suffix = out.suffix
    media = "video/mp4" if suffix == ".mp4" else "image/jpeg"
    return FileResponse(str(out), filename=f"faceswap_{job_id[:8]}{suffix}", media_type=media)


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job não encontrado")
    j = jobs.pop(job_id)
    paths = [j.get(k) for k in ("source_path", "target_path", "output_path")]
    paths += [pr.get("source_path") for pr in j.get("pairs") or []]
    for path in paths:
        try:
            p = Path(path or "")
            if p.is_file():
                p.unlink()
        except Exception:
            pass
    return {"deleted": job_id}


@app.get("/api/status")
async def server_status():
    return {"cuda": HAS_CUDA, "threads": THREAD_COUNT, "max_video_seconds": MAX_VIDEO_SECONDS or None,
            "queue_length": len(pending_ids)}


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
