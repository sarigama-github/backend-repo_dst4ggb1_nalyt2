import os
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from database import db, create_document, get_documents
from schemas import VoiceSample, Consent, Voice, Generation

from bson import ObjectId
import wave
import struct
import math

app = FastAPI(title="AI Voice Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure media directories exist
MEDIA_ROOT = os.path.join(os.getcwd(), "media")
SAMPLES_DIR = os.path.join(MEDIA_ROOT, "samples")
OUTPUTS_DIR = os.path.join(MEDIA_ROOT, "outputs")
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Serve media files
app.mount("/media", StaticFiles(directory=MEDIA_ROOT), name="media")


@app.get("/")
def read_root():
    return {"message": "AI Voice Studio backend ready"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "❌ Not Set" if not os.getenv("DATABASE_URL") else "✅ Set",
        "database_name": "❌ Not Set" if not os.getenv("DATABASE_NAME") else "✅ Set",
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()[:10]
            response["database"] = "✅ Connected & Working"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"

    return response


# -----------------------------
# Templates (static list)
# -----------------------------
class Template(BaseModel):
    id: str
    name: str
    gender: Optional[str] = None
    locale: Optional[str] = None
    style: Optional[str] = None


TEMPLATES: List[Template] = [
    Template(id="t1", name="Calm Neutral", gender="neutral", locale="en-US", style="narration"),
    Template(id="t2", name="Warm Hindi", gender="female", locale="hi-IN", style="conversational"),
    Template(id="t3", name="Crisp English", gender="male", locale="en-IN", style="presenter"),
]


@app.get("/voices/templates", response_model=List[Template])
def list_templates():
    return TEMPLATES


# -----------------------------
# Upload sample + create voice record
# -----------------------------
@app.post("/voices/sample-upload")
async def upload_sample(
    user_name: str = Form(...),
    user_email: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".aac")):
        raise HTTPException(status_code=400, detail="Please upload an audio file (wav/mp3/m4a/aac)")

    # Save file
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    safe_email = user_email.replace("@", "_at_").replace("/", "_")
    save_name = f"{safe_email}-{ts}-{file.filename}"
    save_path = os.path.join(SAMPLES_DIR, save_name)

    with open(save_path, "wb") as out:
        out.write(await file.read())

    sample_doc = VoiceSample(user_name=user_name, user_email=user_email, file_path=f"/media/samples/{save_name}")
    sample_id = create_document("voicesample", sample_doc)

    # Create a pending voice tied to this sample
    voice_doc = Voice(
        user_name=user_name,
        user_email=user_email,
        name=f"{user_name.split(' ')[0]}'s Voice",
        type="cloned",
        source_sample_path=sample_doc.file_path,
        status="pending",
        consent_granted=False,
    )
    voice_id = create_document("voice", voice_doc)

    return {"sample_id": sample_id, "voice_id": voice_id, "sample_url": sample_doc.file_path, "status": "pending"}


# -----------------------------
# Consent
# -----------------------------
class ConsentIn(BaseModel):
    voice_id: str
    attested_by: str
    attested_email: str
    attestation_text: str
    granted: bool = True


@app.post("/voices/consent")
def submit_consent(payload: ConsentIn):
    # Save consent record
    consent_doc = Consent(
        voice_id=payload.voice_id,
        attested_by=payload.attested_by,
        attested_email=payload.attested_email,
        attestation_text=payload.attestation_text,
        granted=payload.granted,
    )
    consent_id = create_document("consent", consent_doc)

    # Update voice to mark consent and ready
    try:
        db.voice.update_one({"_id": ObjectId(payload.voice_id)}, {"$set": {"consent_granted": True, "status": "ready"}})
    except Exception:
        pass

    return {"consent_id": consent_id, "voice_id": payload.voice_id, "status": "ready"}


# -----------------------------
# Generate TTS (placeholder tone synthesis + watermark)
# -----------------------------
class TTSRequest(BaseModel):
    text: str
    voice_mode: str  # "cloned" | "template"
    user_email: Optional[str] = None
    voice_id: Optional[str] = None
    template_id: Optional[str] = None
    requested_format: str = "wav"  # "wav" | "mp3" | "wav-192"


@app.post("/tts/generate")
def generate_tts(req: TTSRequest):
    if not req.text or len(req.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text is required")

    # Validate selection
    if req.voice_mode == "cloned":
        if not req.voice_id:
            raise HTTPException(status_code=400, detail="voice_id required for cloned mode")
        v = db.voice.find_one({"_id": ObjectId(req.voice_id)})
        if not v or not v.get("consent_granted"):
            raise HTTPException(status_code=400, detail="Voice not found or consent not granted")
    elif req.voice_mode == "template":
        if not req.template_id or req.template_id not in [t.id for t in TEMPLATES]:
            raise HTTPException(status_code=400, detail="Valid template_id required")
    else:
        raise HTTPException(status_code=400, detail="Invalid voice_mode")

    # Generate a short tone based on text length (placeholder synthesis)
    duration_sec = min(10.0, max(2.0, len(req.text) / 30.0))
    sample_rate = 44100
    n_samples = int(duration_sec * sample_rate)

    # Two-tone watermarking pattern encoded into samples subtly
    f_base = 440.0  # A4
    f_mark = 523.25  # C5 watermark tone

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    base_name = f"tts-{ts}-{ObjectId()}"

    # Decide format/filename
    ext = "wav" if req.requested_format.startswith("wav") else "mp3"
    file_name = f"{base_name}.{ext}"
    out_path = os.path.join(OUTPUTS_DIR, file_name)

    # WAV synthesis
    with wave.open(out_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            t = i / sample_rate
            # base voice tone + low-amplitude watermark tone
            sample = 0.4 * math.sin(2 * math.pi * f_base * t) + 0.05 * math.sin(2 * math.pi * f_mark * t)
            # simple envelope to avoid click
            env = min(1.0, i / 500.0) * min(1.0, (n_samples - i) / 500.0)
            value = int(max(-1.0, min(1.0, sample * env)) * 32767)
            wf.writeframes(struct.pack('<h', value))

    public_url = f"/media/outputs/{file_name}"

    watermark = {
        "generator": "AI Voice Studio (demo)",
        "mode": req.voice_mode,
        "voice_id": req.voice_id,
        "template_id": req.template_id,
        "format": req.requested_format,
        "timestamp": ts,
    }

    gen_doc = Generation(
        user_email=req.user_email or "",
        text=req.text,
        voice_mode=req.voice_mode,
        voice_id=req.voice_id,
        template_id=req.template_id,
        requested_format=req.requested_format,
        output_path=public_url,
        meta={"placeholder": True},
        watermark=watermark,
        duration_sec=duration_sec,
    )
    gen_id = create_document("generation", gen_doc)

    return {
        "id": gen_id,
        "audio_url": public_url,
        "format": ext,
        "duration_sec": duration_sec,
        "watermark": watermark,
    }


# Convenience endpoint to download file with proper headers
@app.get("/tts/download/{file_name}")
def download(file_name: str):
    path = os.path.join(OUTPUTS_DIR, file_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    media_type = "audio/wav" if file_name.endswith(".wav") else "audio/mpeg"
    return FileResponse(path, media_type=media_type, filename=file_name)
