"""
Database Schemas for Voice Studio

Each Pydantic model corresponds to a MongoDB collection (lowercased class name).
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict
from datetime import datetime

class VoiceSample(BaseModel):
    user_name: str
    user_email: str
    file_path: str
    duration_sec: Optional[float] = None
    status: Literal["uploaded", "processed"] = "uploaded"

class Consent(BaseModel):
    voice_id: str
    attested_by: str
    attested_email: str
    attestation_text: str
    granted: bool = True
    recorded_consent_path: Optional[str] = None
    granted_at: datetime = Field(default_factory=datetime.utcnow)

class Voice(BaseModel):
    user_name: str
    user_email: str
    name: str
    type: Literal["cloned", "template"] = "cloned"
    source_sample_path: Optional[str] = None
    status: Literal["pending", "ready"] = "pending"
    consent_granted: bool = False

class Generation(BaseModel):
    user_email: str
    text: str
    voice_mode: Literal["cloned", "template"]
    voice_id: Optional[str] = None
    template_id: Optional[str] = None
    requested_format: Literal["wav", "mp3", "wav-192"] = "wav"
    output_path: Optional[str] = None
    meta: Dict = {}
    watermark: Dict = {}
    duration_sec: Optional[float] = None
