from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient
from influxdb_client.rest import ApiException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

sys.path.append(str(Path(__file__).resolve().parents[1]))
from aivis_client import AivisClient  # noqa: E402

try:  # pragma: no cover - optional dependency
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - optional dependency
    WhisperModel = None

import requests

load_dotenv()

INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")
GPS_WINDOW = os.getenv("GPS_WINDOW", "1m")
ALTITUDE_WINDOW = os.getenv("ALTITUDE_WINDOW", "1m")
ALTITUDE_MEASUREMENT = os.getenv("ALTITUDE_MEASUREMENT", "gps_fix")
ALTITUDE_FIELD = os.getenv("ALTITUDE_FIELD", "altitude_m")

ENABLE_ASSISTANT = os.getenv("ENABLE_ASSISTANT", "1").lower() not in {"0", "false", "off"}
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "turbo")  # whisper large-v3 turbo 系をデフォルトに
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" をデフォルトにして GPU 未整備環境でのクラッシュを防止
STT_REMOTE_URL = os.getenv("STT_REMOTE_URL")
STT_REMOTE_API_KEY = os.getenv("STT_REMOTE_API_KEY")
STT_REMOTE_MODEL = os.getenv("STT_REMOTE_MODEL", WHISPER_MODEL)

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
ASSISTANT_LLM_MODEL = os.getenv("ASSISTANT_LLM_MODEL", "hf.co/alfredplpl/gemma-2-2b-jpn-it-gguf")
ASSISTANT_TEMPERATURE = float(os.getenv("ASSISTANT_TEMPERATURE", "0.3"))

AIVIS_BASE = os.getenv("AIVIS_BASE", "http://localhost:10102")
AIVIS_SPEAKER = int(os.getenv("AIVIS_SPEAKER", "888753760"))

WEATHER_ENABLED = os.getenv("WEATHER_ENABLED", "1").lower() not in {"0", "false", "off"}
WEATHER_LAT = float(os.getenv("WEATHER_LAT", "35.339"))  # 藤沢市近傍
WEATHER_LON = float(os.getenv("WEATHER_LON", "139.49"))
WEATHER_API_URL = os.getenv("WEATHER_API_URL", "https://api.open-meteo.com/v1/forecast")
WEATHER_POLL_SECONDS = int(os.getenv("WEATHER_POLL_SECONDS", "600"))  # 10 分おきに更新

app = FastAPI(title="Local Metrics API")

# 開発向けCORS（同一オリジン配信なら本来は不要だが、残しても可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

class LatestResponse(BaseModel):
    timestamp: Optional[str] = None
    measurement: str
    user_id: str
    range: str
    data: Dict[str, Any]

class GPSResponse(BaseModel):
    timestamp: Optional[str] = None
    measurement: str
    user_id: str
    range: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    data: Dict[str, Any]


# ===== 音声アシスタント基盤 =====
class SpeechToTextBackend:
    """Audio transcription provider interface."""

    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError


class FasterWhisperBackend(SpeechToTextBackend):
    """Local faster-whisper backend."""

    def __init__(self, model_name: str, compute_type: str = "int8", language: Optional[str] = None) -> None:
        if WhisperModel is None:
            raise ImportError(
                "faster-whisper is not installed. Install it with `pip install faster-whisper` "
                "or set STT_REMOTE_URL to use a remote transcription API."
            )
        self.language = language
        # GPU ライブラリ（cuDNN）が無い環境で自動で GPU を掴むとプロセスごと落ちることがある。
        # 明示的にデバイスを指定し、CPU の場合は CT2_FORCE_CPU を設定して CUDA ロードを抑止する。
        device = os.getenv("WHISPER_DEVICE", "cpu")
        if device.lower() == "cpu":
            os.environ.setdefault("CT2_FORCE_CPU", "1")
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(audio_path, beam_size=1, language=self.language)
        text = "".join(segment.text for segment in segments).strip()
        return text


class RemoteWhisperBackend(SpeechToTextBackend):
    """Simple HTTP backend compatible with OpenAI Whisper-like APIs."""

    def __init__(self, endpoint: str, api_key: Optional[str], model_name: str) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.model_name = model_name

    def transcribe(self, audio_path: str) -> str:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with open(audio_path, "rb") as audio_fp:
            files = {"file": (Path(audio_path).name, audio_fp, "audio/wav")}
            data = {"model": self.model_name}
            response = requests.post(
                self.endpoint,
                files=files,
                data=data,
                headers=headers,
                timeout=int(os.getenv("STT_REMOTE_TIMEOUT", "120")),
            )
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, dict):
            if "text" in payload and isinstance(payload["text"], str):
                return payload["text"].strip()
            if "data" in payload and isinstance(payload["data"], list):
                text = " ".join(item.get("text", "") for item in payload["data"])
                return text.strip()

        raise RuntimeError(f"Unexpected STT response: {json.dumps(payload, ensure_ascii=False)}")


def build_transcriber() -> SpeechToTextBackend:
    if STT_REMOTE_URL:
        return RemoteWhisperBackend(STT_REMOTE_URL, STT_REMOTE_API_KEY, STT_REMOTE_MODEL)
    return FasterWhisperBackend(WHISPER_MODEL, WHISPER_COMPUTE_TYPE, language=WHISPER_LANGUAGE)


class AssistantPipeline:
    """High-level pipeline that handles STT -> LLM -> optional TTS."""

    SYSTEM_PROMPT = (
        "あなたは丁寧な日本語アシスタントです。ユーザーの発話を踏まえて、"
        "事実に基づいた返答を1〜2文で簡潔に返してください。"
        "ユーザが近くの道路状況について尋ねた場合は、天気の情報を加味して大丈夫そうかを答えてください。"
    )

    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required to run the assistant pipeline.")
        self.transcriber = build_transcriber()
        self.llm = ChatOpenAI(
            model=ASSISTANT_LLM_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=ASSISTANT_TEMPERATURE,
        )
        self.tts: Optional[AivisClient] = AivisClient(AIVIS_BASE, AIVIS_SPEAKER) if AIVIS_BASE else None

    async def process_audio(self, audio_path: Path) -> Dict[str, Any]:
        transcript = await asyncio.to_thread(self.transcriber.transcribe, str(audio_path))
        transcript = transcript.strip()

        if not transcript:
            return {
                "status": "no-speech",
                "message": "音声を認識できませんでした。もう一度お話しください。",
                "transcript": "",
            }

        reply = await self._generate_reply(transcript)
        audio_bytes = await self._speak(reply)
        payload: Dict[str, Any] = {
            "status": "ok",
            "transcript": transcript,
            "reply": reply,
        }
        if audio_bytes:
            payload["audio_base64"] = base64.b64encode(audio_bytes).decode("ascii")
            payload["audio_mime"] = "audio/wav"
        return payload

    async def _generate_reply(self, transcript: str) -> str:
        messages = [SystemMessage(content=self.SYSTEM_PROMPT)]

        weather_ctx = get_weather_context()
        if weather_ctx:
            messages.append(
                SystemMessage(
                    content=(
                        "天気や路面状況に関する質問には、次のスナップショットを根拠に必ず答えてください。"
                        "天気以外の話題ではこの情報を挿入しないでください。\n"
                        f"{weather_ctx}"
                    )
                )
            )

        messages.append(HumanMessage(content=transcript))
        response = await asyncio.to_thread(self.llm.invoke, messages)
        return response.content.strip()

    async def _speak(self, text: str) -> Optional[bytes]:
        if not self.tts:
            return None
        return await asyncio.to_thread(self.tts.tts, text)


assistant_pipeline: Optional[AssistantPipeline] = None
weather_snapshot: Optional[str] = None
weather_updated_at: Optional[str] = None
weather_lock = asyncio.Lock()


# ===== 天気スナップショット (藤沢市 直近1時間) =====
def _describe_weather_code(code: int) -> str:
    mapping = {
        0: "快晴",
        1: "ほぼ快晴",
        2: "薄曇り",
        3: "曇り",
        45: "霧",
        48: "霧（霧氷）",
        51: "霧雨（弱い）",
        53: "霧雨（中）",
        55: "霧雨（強い）",
        56: "着氷霧雨（弱い）",
        57: "着氷霧雨（強い）",
        61: "雨（弱い）",
        63: "雨（中）",
        65: "雨（強い）",
        66: "着氷雨（弱い）",
        67: "着氷雨（強い）",
        71: "雪（弱い）",
        73: "雪（中）",
        75: "雪（強い）",
        77: "雪の粒",
        80: "にわか雨（弱い）",
        81: "にわか雨（中）",
        82: "にわか雨（強い）",
        85: "にわか雪（弱い）",
        86: "にわか雪（強い）",
        95: "雷雨（弱〜中）",
        96: "雷雨（ひょうを伴う弱〜中）",
        99: "雷雨（ひょうを伴う強い）",
    }
    return mapping.get(int(code), f"不明な天気コード({code})")


def _build_weather_summary(payload: Dict[str, Any]) -> Tuple[str, str]:
    """Return (summary, time_iso)."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        raise ValueError("hourly time is missing")
    idx = 0  # 直近1時間のみを使う
    time_iso = times[idx]
    code = (hourly.get("weathercode") or [None])[idx]
    temp = (hourly.get("temperature_2m") or [None])[idx]
    precip = (hourly.get("precipitation") or [0])[idx] or 0
    rain = (hourly.get("rain") or [0])[idx] or 0
    showers = (hourly.get("showers") or [0])[idx] or 0
    snow = (hourly.get("snowfall") or [0])[idx] or 0
    wind = (hourly.get("windspeed_10m") or [None])[idx]
    direction = (hourly.get("winddirection_10m") or [None])[idx]

    code_desc = _describe_weather_code(code if code is not None else -1)
    time_obj = datetime.fromisoformat(time_iso)
    if time_obj.tzinfo is None:
        time_obj = time_obj.replace(tzinfo=timezone.utc)
    time_obj = time_obj.astimezone()
    hhmm = time_obj.strftime("%H:%M")

    precip_mm = float(precip)
    rain_mm = float(rain)
    showers_mm = float(showers)
    snow_mm = float(snow)
    wetness = precip_mm + rain_mm + showers_mm + snow_mm
    if wetness > 0.1:
        road = "雨で路面が濡れている可能性があります。速度に注意してください。"
    elif code in {71, 73, 75, 77, 85, 86}:
        road = "雪の恐れがあります。滑りやすい路面に注意してください。"
    elif code in {45, 48}:
        road = "霧で視界が悪いかもしれません。"
    else:
        road = "路面はおおむね乾いていて走行しやすそうです。"

    temp_text = f"{temp:.1f}℃" if temp is not None else "N/A"
    wind_text = f"{wind:.1f}m/s" if wind is not None else "不明"
    dir_text = f"{int(direction)}°" if direction is not None else "不明"
    summary = (
        f"藤沢市の直近1時間予報（{hhmm}頃まで）: {code_desc}, 気温 {temp_text}, "
        f"降水量 {precip_mm:.1f}mm, 風速 {wind_text} (方角 {dir_text})。路面: {road}"
    )
    return summary, time_iso


async def fetch_weather_once() -> Tuple[str, str]:
    params = {
        "latitude": WEATHER_LAT,
        "longitude": WEATHER_LON,
        "hourly": ",".join([
            "temperature_2m",
            "precipitation",
            "rain",
            "showers",
            "snowfall",
            "weathercode",
            "windspeed_10m",
            "winddirection_10m",
        ]),
        "forecast_days": 1,
        "timezone": "Asia/Tokyo",
    }
    resp = requests.get(WEATHER_API_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return _build_weather_summary(data)


async def refresh_weather_periodically() -> None:
    global weather_snapshot, weather_updated_at
    while True:
        try:
            summary, ts = await fetch_weather_once()
            async with weather_lock:
                weather_snapshot = summary
                weather_updated_at = ts
        except Exception as exc:  # noqa: BLE001
            # ログだけ残して次回に再試行
            print(f"[weather] fetch failed: {exc}")
        await asyncio.sleep(max(60, WEATHER_POLL_SECONDS))


def get_weather_context() -> Optional[str]:
    """Return latest weather snapshot text."""
    return weather_snapshot


@app.on_event("startup")
async def init_assistant() -> None:
    global assistant_pipeline
    if not ENABLE_ASSISTANT:
        print("[assistant] disabled via ENABLE_ASSISTANT env")
        return
    try:
        assistant_pipeline = AssistantPipeline()
        print("[assistant] pipeline initialized")
    except Exception as exc:  # noqa: BLE001
        assistant_pipeline = None
        print(f"[assistant] initialization failed: {exc}")

    if WEATHER_ENABLED:
        asyncio.create_task(refresh_weather_periodically())
        print("[weather] updater started")

def build_flux_query(bucket: str, time_range: str, user_id: str,
                     measurement: str, fields: Optional[List[str]]):
    fld_filter = " or ".join([f'r._field == "{f}"' for f in fields]) if fields else "true"
    flux = f'''
from(bucket: "{bucket}")
  |> range(start: {time_range})
  |> filter(fn: (r) => r.user_id == "{user_id}")
  |> filter(fn: (r) => r._measurement == "{measurement}")
  |> filter(fn: (r) => {fld_filter})
  |> group(columns: ["_field"])
  |> sort(columns: ["_time"], desc: true)
  |> unique(column: "_field")
'''
    return flux

def round_half_up(value: float, digits: int = 1) -> float:
    """Round using the expected 四捨五入 semantics."""
    quant = Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))

def build_gps_flux_query(bucket: str, time_range: str, user_id: str,
                         measurement: str, fields: Optional[List[str]],
                         window: str = "1m") -> str:
    fld_list = fields or ["latitude", "longitude"]
    fld_filter = " or ".join([f'r["_field"] == "{f}"' for f in fld_list])
    keep_columns = ", ".join([f'"{col}"' for col in ["_time", *fld_list]])
    flux = f'''
from(bucket: "{bucket}")
  |> range(start: {time_range})
  |> filter(fn: (r) => r["user_id"] == "{user_id}")
  |> filter(fn: (r) => r["_measurement"] == "{measurement}")
  |> filter(fn: (r) => {fld_filter})
  |> aggregateWindow(every: {window}, fn: last, createEmpty: false)
  |> group(columns: ["_measurement", "user_id"], mode: "by")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: [{keep_columns}])
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 1)
'''
    return flux

def build_altitude_flux_query(bucket: str, time_range: str, user_id: str,
                              measurement: str = "gps_fix", field: str = "altitude_m",
                              window: str = "1m") -> str:
    flux = f'''
from(bucket: "{bucket}")
  |> range(start: {time_range})
  |> filter(fn: (r) => r["user_id"] == "{user_id}")
  |> filter(fn: (r) => r["_field"] == "{field}")
  |> filter(fn: (r) => r["_measurement"] == "{measurement}")
  |> aggregateWindow(every: {window}, fn: mean, createEmpty: false)
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 1)
'''
    return flux

@app.get("/api/latest", response_model=LatestResponse)
async def latest(
    user_id: str = Query(..., description="Influx の user_id タグ"),
    measurement: str = Query("bsec_air_quality"),
    fields: Optional[str] = Query("temperature_c,humidity_percent,altitude_m"),
    time_range: str = Query("-1h"),
):
    fld_list = [f.strip() for f in fields.split(",")] if fields else None
    try:
        with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
            q = build_flux_query(INFLUX_BUCKET, time_range, user_id, measurement, fld_list)
            tables = client.query_api().query(org=INFLUX_ORG, query=q)
    except ApiException as exc:
        raise HTTPException(status_code=502, detail=f"Influx query failed: {exc.status} {exc.reason}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=502, detail=f"Influx query failed: {exc}") from exc

    latest_ts = None
    result: Dict[str, Any] = {}
    for table in tables:
        for record in table.records:
            fld = record.get_field()
            val = record.get_value()
            ts = record.get_time().isoformat()
            result[fld] = val
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts

    needs_altitude = (fld_list is None) or ("altitude_m" in (fld_list or []))
    if needs_altitude:
        altitude_value: Optional[float] = None
        altitude_ts: Optional[str] = None
        alt_query = build_altitude_flux_query(
            INFLUX_BUCKET,
            time_range,
            user_id,
            measurement=ALTITUDE_MEASUREMENT,
            field=ALTITUDE_FIELD,
            window=ALTITUDE_WINDOW,
        )
        with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
            alt_tables = client.query_api().query(org=INFLUX_ORG, query=alt_query)
        for table in alt_tables:
            for record in table.records:
                value = record.get_value()
                if value is None:
                    continue
                altitude_value = float(value)
                altitude_ts = record.get_time().isoformat()
                break
            if altitude_value is not None:
                break
        if altitude_value is not None:
            result["altitude_m"] = round_half_up(altitude_value, 1)
            if altitude_ts and (latest_ts is None or altitude_ts > latest_ts):
                latest_ts = altitude_ts

    return LatestResponse(
        timestamp=latest_ts,
        measurement=measurement,
        user_id=user_id,
        range=time_range,
        data=result,
    )

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/api/latest_gps", response_model=GPSResponse)
async def latest_gps(
    user_id: str = Query(..., description="Influx の user_id タグ"),
    measurement: str = Query("gps_fix"),
    fields: Optional[str] = Query("latitude,longitude"),
    time_range: str = Query("-1h"),
):
    fld_list = [f.strip() for f in fields.split(",")] if fields else None
    effective_fields = fld_list or ["latitude", "longitude"]
    try:
        with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
            q = build_gps_flux_query(
                INFLUX_BUCKET,
                time_range,
                user_id,
                measurement,
                effective_fields,
                window=GPS_WINDOW,
            )
            tables = client.query_api().query(org=INFLUX_ORG, query=q)
    except ApiException as exc:
        raise HTTPException(status_code=502, detail=f"Influx query failed: {exc.status} {exc.reason}") from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"Influx query failed: {exc}") from exc

    latest_ts = None
    lat = None
    lon = None
    result: Dict[str, Any] = {}
    def update_latest(ts: Optional[str]) -> None:
        nonlocal latest_ts
        if ts is None:
            return
        if latest_ts is None or ts > latest_ts:
            latest_ts = ts

    def coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def assign_value(field: str, raw_value: Any, ts: Optional[str]) -> None:
        nonlocal lat, lon
        if raw_value is None:
            return
        numeric = coerce_float(raw_value)
        result[field] = numeric if numeric is not None else raw_value
        if field == "latitude" and lat is None and numeric is not None:
            lat = numeric
            update_latest(ts)
        elif field == "longitude" and lon is None and numeric is not None:
            lon = numeric
            update_latest(ts)
        elif field not in {"latitude", "longitude"}:
            update_latest(ts)

    for table in tables:
        for record in table.records:
            ts = record.get_time().isoformat()
            values = record.values
            for fld in effective_fields:
                assign_value(fld, values.get(fld), ts)
            try:
                fld_name = record.get_field()
            except KeyError:
                fld_name = None
            if fld_name in effective_fields:
                assign_value(fld_name, record.get_value(), ts)
            if lat is not None and lon is not None:
                break
        if lat is not None and lon is not None:
            break

    missing_fields = []
    if lat is None and "latitude" in effective_fields:
        missing_fields.append("latitude")
    if lon is None and "longitude" in effective_fields:
        missing_fields.append("longitude")
    if missing_fields:
        with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
            fallback_query = build_flux_query(INFLUX_BUCKET, time_range, user_id, measurement, missing_fields)
            fallback_tables = client.query_api().query(org=INFLUX_ORG, query=fallback_query)
        for table in fallback_tables:
            for record in table.records:
                ts = record.get_time().isoformat()
                fld_name = record.get_field()
                if fld_name in missing_fields:
                    assign_value(fld_name, record.get_value(), ts)
                if lat is not None and lon is not None:
                    break
            if lat is not None and lon is not None:
                break

    return GPSResponse(
        timestamp=latest_ts,
        measurement=measurement,
        user_id=user_id,
        range=time_range,
        lat=lat,
        lon=lon,
        data=result,
    )

@app.post("/api/assistant/talk")
async def assistant_talk(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not ENABLE_ASSISTANT or assistant_pipeline is None:
        raise HTTPException(status_code=503, detail="Assistant pipeline is disabled")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Audio payload is empty")

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = Path(tmp_file.name)
    try:
        tmp_file.write(contents)
        tmp_file.flush()
        tmp_file.close()
        result = await assistant_pipeline.process_audio(tmp_path)
        return result
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Assistant processing failed: {exc}") from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:  # pragma: no cover - best effort cleanup
            pass


class TtsRequest(BaseModel):
    text: str


@app.post("/api/tts")
async def tts_endpoint(payload: TtsRequest = Body(...)) -> Dict[str, Any]:
    """Synthesize speech for arbitrary text using AIVIS TTS if available."""
    if not assistant_pipeline or not assistant_pipeline.tts:
        raise HTTPException(status_code=503, detail="TTS is not available")
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    try:
        audio_bytes = await asyncio.to_thread(assistant_pipeline.tts.tts, text)
        return {
            "status": "ok",
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
            "audio_mime": "audio/wav",
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc

# ===== 静的ファイル（フロントエンド）を同一オリジンで配信 =====
# このファイルと同じディレクトリに public/ を作り、そこへ index.html を置く
static_dir = Path(__file__).parent / "public"
static_dir.mkdir(exist_ok=True)
# ルート（/）にマウントすることで http://<PCのIP>:8000/ で index.html を配信
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
