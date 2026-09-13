from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path

from dotenv import load_dotenv
from vosk import KaldiRecognizer
from vosk import Model


MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_LANGUAGE = "en-us"
DEFAULT_MODELS_DIR = "models"


class VoskUnavailableError(RuntimeError):
    """Raised when the Vosk model is unavailable or fails to load."""

    pass


def load_client() -> Model:
    load_dotenv()

    models_dir = Path(
        os.getenv(
            "VOSK_MODELS_DIR",
            DEFAULT_MODELS_DIR,
        )
    )

    model_path = models_dir / MODEL_NAME

    if not model_path.exists():
        raise VoskUnavailableError(
            f"Vosk model not found at: {model_path}. "
            "Download it from https://alphacephei.com/vosk/models "
            "or set VOSK_MODELS_DIR in your .env file."
        )

    print("Loading Vosk model...")

    try:
        model = Model(
            str(model_path)
        )

    except Exception as exc:
        raise VoskUnavailableError(
            "Vosk model failed to load. "
            "Please try the transcription again."
        ) from exc

    print("Vosk model loaded successfully.")

    return model


def upload_audio(
    model: Model,
    audio_path: Path,
):
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print("Loading audio for Vosk...")

    try:
        audio_file = wave.open(
            str(audio_path),
            "rb",
        )

    except wave.Error as exc:
        raise RuntimeError(
            "Vosk requires 16-bit PCM WAV audio. "
            "Convert the file first, e.g. with ffmpeg."
        ) from exc

    print("Audio loaded successfully.")

    return audio_file


def _parse_segment(
    result: dict,
):

    text = str(
        result.get("text", "")
    ).strip()

    if not text:
        return None

    words = result.get(
        "result"
    ) or []

    if words:
        start = float(
            words[0]["start"]
        )

        end = float(
            words[-1]["end"]
        )

    else:
        start = 0.0
        end = 0.0

    return {
        "start": start,
        "end": end,
        "text": text,
    }


def transcribe_audio(
    model: Model,
    audio_file,
) -> dict:

    recognizer = KaldiRecognizer(
        model,
        audio_file.getframerate(),
    )

    recognizer.SetWords(True)

    segments = []

    try:
        while True:
            data = audio_file.readframes(
                4000
            )

            if len(data) == 0:
                break

            if recognizer.AcceptWaveform(
                data
            ):
                segment = _parse_segment(
                    json.loads(
                        recognizer.Result()
                    )
                )

                if segment:
                    segments.append(
                        segment
                    )

        segment = _parse_segment(
            json.loads(
                recognizer.FinalResult()
            )
        )

        if segment:
            segments.append(
                segment
            )

    finally:
        audio_file.close()

    return {
        "language": MODEL_LANGUAGE,
        "segments": segments,
    }


def validate_transcript(
    result: dict,
) -> dict:

    if not isinstance(result, dict):
        raise ValueError(
            "Transcript must be a JSON object."
        )

    if "segments" not in result:
        raise ValueError(
            "Transcript response does not contain "
            "'segments'."
        )

    if not isinstance(
        result["segments"],
        list,
    ):
        raise ValueError(
            "'segments' must be a list."
        )

    validated_segments = []

    for segment in result["segments"]:

        if not isinstance(
            segment,
            dict,
        ):
            continue

        if not all(
            key in segment
            for key in (
                "start",
                "end",
                "text",
            )
        ):
            continue

        try:
            start = float(
                segment["start"]
            )
            end = float(
                segment["end"]
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        text = str(
            segment["text"]
        ).strip()

        if not text:
            continue

        if start < 0:
            continue

        if end < start:
            continue

        validated_segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
            }
        )

    validated_segments.sort(
        key=lambda segment:
        segment["start"]
    )

    return {
        "language": result.get(
            "language"
        ),
        "segments": validated_segments,
    }


def print_transcript(
    result: dict,
) -> None:

    print()
    print("=" * 70)
    print("TRANSCRIPT")
    print("=" * 70)

    language = result.get(
        "language"
    )

    if language:
        print(
            f"Language: {language}"
        )

    print()

    segments = result[
        "segments"
    ]

    if not segments:
        print(
            "No speech detected."
        )
        return

    for index, segment in enumerate(
        segments,
        start=1,
    ):
        print(
            f"[{index:02d}] "
            f"{segment['start']:07.2f}s → "
            f"{segment['end']:07.2f}s"
        )

        print(
            f"     {segment['text']}"
        )

        print()


def main() -> None:

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            '  uv run python -m voice_id.transcription '
            '"path/to/audio"'
        )

        raise SystemExit(1)

    audio_path = Path(
        sys.argv[1]
    )

    print("=" * 70)
    print("VOICE TRANSCRIPTION")
    print("=" * 70)

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Audio: {audio_path}"
    )

    try:
        client = load_client()

        audio_file = upload_audio(
            client,
            audio_path,
        )

        print()
        print("Transcribing...")

        result = transcribe_audio(
            client,
            audio_file,
        )

        result = validate_transcript(
            result
        )

        print_transcript(
            result
        )

        print("=" * 70)
        print("DONE")
        print("=" * 70)

    except VoskUnavailableError as exc:
        print()
        print("=" * 70)
        print("TRANSCRIPTION UNAVAILABLE")
        print("=" * 70)
        print()
        print(str(exc))
        print()

        raise SystemExit(2)

    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        print()
        print("=" * 70)
        print("TRANSCRIPTION FAILED")
        print("=" * 70)
        print()
        print(str(exc))
        print()

        raise SystemExit(1)


if __name__ == "__main__":
    main()