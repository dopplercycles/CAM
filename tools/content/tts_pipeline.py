"""
CAM TTS Voice Pipeline

Piper-based text-to-speech with graceful fallback when Piper isn't installed.
Mirrors the LongTermMemory pattern: lazy initialization, clear error messages,
full functionality once the dependency is available.

Piper is a fast, local neural TTS engine. Voice models are ONNX files stored
in data/audio/voices/. Audio output is WAV files in data/audio/ with JSON
sidecar metadata.

Usage:
    from tools.content.tts_pipeline import TTSPipeline

    tts = TTSPipeline()
    print(tts.get_status())          # shows "piper binary not found" if missing
    result = await tts.synthesize("Hello from Cam!")
    if result.error:
        print(f"TTS failed: {result.error}")
    else:
        print(f"Audio saved: {result.audio_path} ({result.duration_secs:.1f}s)")
"""

import asyncio
import json
import logging
import os
import shutil
import wave
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("cam.tts")


# ---------------------------------------------------------------------------
# SynthesisResult — returned by every synthesize() call
# ---------------------------------------------------------------------------

@dataclass
class SynthesisResult:
    """Result of a TTS synthesis attempt.

    Always returned — check error field to determine success/failure.
    On success: audio_path is set, duration_secs > 0, error is None.
    On failure: audio_path is None, error describes what went wrong.
    """

    audio_path: str | None      # path to WAV file, or None on failure
    text: str                   # source text that was synthesized
    voice: str                  # voice model used (or requested)
    duration_secs: float        # audio duration in seconds (0 if failed)
    timestamp: str              # ISO UTC timestamp
    error: str | None           # error message if synthesis failed

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# TTSPipeline — lazy-init Piper TTS with graceful fallback
# ---------------------------------------------------------------------------

class TTSPipeline:
    """Piper TTS pipeline with graceful degradation.

    Works immediately once Piper is installed and a voice model is placed
    in the voices directory. Until then, all methods return clear error
    messages instead of crashing.

    Args:
        config: Optional CAMConfig instance. If None, uses sensible defaults.
    """

    def __init__(self, config=None):
        # Read config with safe attribute access (matches server.py pattern)
        tts_cfg = getattr(config, 'tts', None) if config else None

        self._audio_dir = Path(
            getattr(tts_cfg, 'audio_dir', None) or "data/audio"
        )
        self._voices_dir = Path(
            getattr(tts_cfg, 'voices_dir', None) or "data/audio/voices"
        )
        self._default_voice = (
            getattr(tts_cfg, 'default_voice', None) or "en_US-lessac-medium"
        )
        self._sample_rate = int(
            getattr(tts_cfg, 'sample_rate', None) or 22050
        )

        # Lazy init state
        self._initialized: bool = False
        self._error: str | None = None
        self._piper_binary: str | None = None
        self._current_voice: str = self._default_voice

        logger.info(
            "TTSPipeline created (audio_dir=%s, voices_dir=%s, default_voice=%s)",
            self._audio_dir, self._voices_dir, self._default_voice,
        )

    # -------------------------------------------------------------------
    # Lazy initialization
    # -------------------------------------------------------------------

    def _ensure_initialized(self) -> bool:
        """Check for Piper binary and voice models.

        Creates directories if needed. Sets _initialized=True only if
        Piper is found. Records clear error message otherwise.

        Returns:
            True if ready for synthesis, False if not.
        """
        if self._initialized:
            return True

        # Ensure directories exist
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        self._voices_dir.mkdir(parents=True, exist_ok=True)

        # Look for piper binary
        piper_path = shutil.which("piper")
        if not piper_path:
            self._error = (
                "piper binary not found — install Piper TTS and ensure "
                "'piper' is on PATH. See: https://github.com/rhasspy/piper"
            )
            logger.warning("TTS init: %s", self._error)
            return False

        self._piper_binary = piper_path
        self._error = None
        self._initialized = True
        logger.info("TTS initialized: piper=%s", piper_path)
        return True

    # -------------------------------------------------------------------
    # Core synthesis
    # -------------------------------------------------------------------

    async def synthesize(
        self, text: str, voice: str | None = None, filename: str | None = None
    ) -> SynthesisResult:
        """Synthesize text to a WAV file using Piper.

        Args:
            text:     Text to synthesize.
            voice:    Voice model name (without .onnx). Defaults to current voice.
            filename: Output filename (without path). Auto-generated if None.

        Returns:
            SynthesisResult with audio_path on success, error on failure.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        voice = voice or self._current_voice

        if not text or not text.strip():
            return SynthesisResult(
                audio_path=None, text=text or "", voice=voice,
                duration_secs=0, timestamp=timestamp,
                error="Empty text — nothing to synthesize",
            )

        if not self._ensure_initialized():
            return SynthesisResult(
                audio_path=None, text=text, voice=voice,
                duration_secs=0, timestamp=timestamp, error=self._error,
            )

        # Resolve voice model path
        voice_model = self._voices_dir / f"{voice}.onnx"
        if not voice_model.exists():
            available = self.list_voices()
            hint = f" Available: {', '.join(available)}" if available else ""
            error = f"Voice model not found: {voice_model}.{hint}"
            return SynthesisResult(
                audio_path=None, text=text, voice=voice,
                duration_secs=0, timestamp=timestamp, error=error,
            )

        # Generate output filename
        if not filename:
            ts_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            # Sanitize first few words for a readable filename
            words = "".join(c if c.isalnum() or c == " " else "" for c in text[:40])
            slug = "_".join(words.split()[:5]).lower()
            filename = f"{ts_slug}_{slug}.wav"

        # Ensure filename ends with .wav
        if not filename.endswith(".wav"):
            filename += ".wav"

        output_path = self._audio_dir / filename

        # Run piper via subprocess: echo text | piper --model voice.onnx --output_file out.wav
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._run_piper, text, str(voice_model), str(output_path))

            if result is not None:
                # result is an error string
                return SynthesisResult(
                    audio_path=None, text=text, voice=voice,
                    duration_secs=0, timestamp=timestamp, error=result,
                )

        except Exception as e:
            error = f"Piper execution failed: {e}"
            logger.error("TTS synthesis error: %s", error)
            return SynthesisResult(
                audio_path=None, text=text, voice=voice,
                duration_secs=0, timestamp=timestamp, error=error,
            )

        # Read WAV duration from header
        duration_secs = self._get_wav_duration(str(output_path))

        # Get file size
        try:
            file_size = output_path.stat().st_size
        except OSError:
            file_size = 0

        # Write JSON sidecar metadata
        sidecar = {
            "text": text,
            "voice": voice,
            "timestamp": timestamp,
            "duration_secs": round(duration_secs, 2),
            "sample_rate": self._sample_rate,
            "file_size_bytes": file_size,
        }
        sidecar_path = output_path.with_suffix(".json")
        try:
            sidecar_path.write_text(json.dumps(sidecar, indent=2))
        except OSError as e:
            logger.warning("Failed to write TTS sidecar %s: %s", sidecar_path, e)

        logger.info(
            "TTS synthesis complete: %s (%.1fs, %d bytes, voice=%s)",
            filename, duration_secs, file_size, voice,
        )

        return SynthesisResult(
            audio_path=str(output_path),
            text=text,
            voice=voice,
            duration_secs=round(duration_secs, 2),
            timestamp=timestamp,
            error=None,
        )

    def _run_piper(self, text: str, model_path: str, output_path: str) -> str | None:
        """Run piper subprocess synchronously (called via run_in_executor).

        Args:
            text:        Text to synthesize.
            model_path:  Path to the .onnx voice model.
            output_path: Path for the output WAV file.

        Returns:
            None on success, error string on failure.
        """
        import subprocess

        try:
            proc = subprocess.run(
                [self._piper_binary, "--model", model_path, "--output_file", output_path],
                input=text,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.strip()[:200] if proc.stderr else "unknown error"
                return f"Piper exited with code {proc.returncode}: {stderr}"
            return None
        except subprocess.TimeoutExpired:
            return "Piper timed out after 60 seconds"
        except FileNotFoundError:
            self._initialized = False
            self._error = "piper binary disappeared from PATH"
            return self._error

    @staticmethod
    def _get_wav_duration(wav_path: str) -> float:
        """Read WAV duration from file header using stdlib wave module.

        Args:
            wav_path: Path to the WAV file.

        Returns:
            Duration in seconds, or 0 if the file can't be read.
        """
        try:
            with wave.open(wav_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return frames / rate
        except Exception as e:
            logger.warning("Failed to read WAV duration from %s: %s", wav_path, e)
        return 0.0

    # -------------------------------------------------------------------
    # Voice management
    # -------------------------------------------------------------------

    def set_voice(self, voice_name: str) -> str | None:
        """Set the current voice for synthesis.

        Validates that the .onnx model file exists in voices_dir.

        Args:
            voice_name: Voice model name (without .onnx extension).

        Returns:
            None on success, error string if voice model not found.
        """
        self._voices_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._voices_dir / f"{voice_name}.onnx"
        if not model_path.exists():
            available = self.list_voices()
            hint = f" Available: {', '.join(available)}" if available else ""
            return f"Voice model not found: {model_path}.{hint}"

        self._current_voice = voice_name
        logger.info("TTS voice set to: %s", voice_name)
        return None

    def list_voices(self) -> list[str]:
        """List available voice models in the voices directory.

        Returns:
            List of voice names (without .onnx extension), sorted alphabetically.
        """
        self._voices_dir.mkdir(parents=True, exist_ok=True)
        voices = []
        for f in sorted(self._voices_dir.glob("*.onnx")):
            voices.append(f.stem)
        return voices

    # -------------------------------------------------------------------
    # Queue synthesis (batch)
    # -------------------------------------------------------------------

    async def queue_synthesis(
        self, items: list[dict[str, str]]
    ) -> list[SynthesisResult]:
        """Synthesize multiple items sequentially.

        Args:
            items: List of dicts with "text" and optional "voice" keys.

        Returns:
            List of SynthesisResult objects, one per item.
        """
        results = []
        for item in items:
            text = item.get("text", "")
            voice = item.get("voice")
            result = await self.synthesize(text=text, voice=voice)
            results.append(result)
        return results

    # -------------------------------------------------------------------
    # Audio file management
    # -------------------------------------------------------------------

    def list_audio(self) -> list[dict[str, Any]]:
        """List all generated audio files with their metadata.

        Reads sidecar JSON for each WAV file. Files without sidecars
        get minimal metadata from the filesystem.

        Returns:
            List of metadata dicts, sorted newest-first by timestamp.
        """
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        audio_files = []

        for wav_path in self._audio_dir.glob("*.wav"):
            sidecar_path = wav_path.with_suffix(".json")
            if sidecar_path.exists():
                try:
                    metadata = json.loads(sidecar_path.read_text())
                except (json.JSONDecodeError, OSError):
                    metadata = {}
            else:
                metadata = {}

            # Ensure filename is always present
            metadata["filename"] = wav_path.name

            # Fill in missing fields from filesystem
            if "file_size_bytes" not in metadata:
                try:
                    metadata["file_size_bytes"] = wav_path.stat().st_size
                except OSError:
                    metadata["file_size_bytes"] = 0

            if "duration_secs" not in metadata:
                metadata["duration_secs"] = self._get_wav_duration(str(wav_path))

            if "timestamp" not in metadata:
                try:
                    mtime = wav_path.stat().st_mtime
                    metadata["timestamp"] = datetime.fromtimestamp(
                        mtime, tz=timezone.utc
                    ).isoformat()
                except OSError:
                    metadata["timestamp"] = ""

            audio_files.append(metadata)

        # Sort newest-first
        audio_files.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return audio_files

    def delete_audio(self, filename: str) -> str | None:
        """Delete an audio file and its sidecar metadata.

        Includes path traversal protection — filename must resolve
        to a path within audio_dir.

        Args:
            filename: WAV filename to delete (just the filename, no path).

        Returns:
            None on success, error string on failure.
        """
        self._audio_dir.mkdir(parents=True, exist_ok=True)

        # Path traversal protection
        target = (self._audio_dir / filename).resolve()
        audio_dir_resolved = self._audio_dir.resolve()

        if not str(target).startswith(str(audio_dir_resolved)):
            logger.warning("TTS delete blocked — path traversal attempt: %s", filename)
            return "Invalid filename — path traversal not allowed"

        if not target.exists():
            return f"File not found: {filename}"

        if not target.suffix == ".wav":
            return "Only .wav files can be deleted"

        # Delete WAV file
        try:
            target.unlink()
        except OSError as e:
            return f"Failed to delete {filename}: {e}"

        # Delete sidecar JSON if it exists
        sidecar = target.with_suffix(".json")
        if sidecar.exists():
            try:
                sidecar.unlink()
            except OSError:
                logger.warning("Failed to delete sidecar %s", sidecar)

        logger.info("TTS audio deleted: %s", filename)
        return None

    # -------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get current TTS pipeline status.

        Returns a dict suitable for JSON serialization and dashboard display.
        """
        # Run init check to get current state
        self._ensure_initialized()

        voice_count = len(self.list_voices())
        audio_count = len(list(self._audio_dir.glob("*.wav"))) if self._audio_dir.exists() else 0

        return {
            "initialized": self._initialized,
            "error": self._error,
            "piper_binary": self._piper_binary,
            "default_voice": self._default_voice,
            "current_voice": self._current_voice,
            "voice_count": voice_count,
            "audio_count": audio_count,
            "audio_dir": str(self._audio_dir),
            "voices_dir": str(self._voices_dir),
            "sample_rate": self._sample_rate,
        }
