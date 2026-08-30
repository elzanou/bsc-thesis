import hashlib
import logging
import math
import struct
import subprocess
import time
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3  # Total attempts (2 retries)
RETRY_DELAY = 1.0  # Initial delay in seconds (doubles each retry)

# Separator tone: 1s silence + 1s tone + 1s silence (3s total)
SEPARATOR_FREQ = 1000  # Hz — standard calibration tone, avoids musical pitches
SEPARATOR_TONE_DURATION = 1.0  # seconds
SEPARATOR_SILENCE_DURATION = 1.0  # seconds
SEPARATOR_AMPLITUDE_RATIO = 1.414  # sqrt(2) — beep RMS matches audio RMS


class AudioProcessor:
    """Handles audio downloading, processing, and concatenation."""

    def __init__(
        self,
        student_audio_dir: Path,
        youtube_cache_dir: Path,
        sample_rate: int = 48000,
    ):
        """Initialize audio processor.

        Args:
            student_audio_dir: Directory containing student audio files
            youtube_cache_dir: Directory for caching YouTube downloads
            sample_rate: Target sample rate for all output (default: 48kHz)
        """
        self.student_audio_dir = student_audio_dir
        self.youtube_cache_dir = youtube_cache_dir
        self.sample_rate = sample_rate
        self.youtube_cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _measure_rms(audio_path: Path) -> float:
        """Measure RMS amplitude of a WAV file (16-bit mono assumed after resampling)."""
        with wave.open(str(audio_path), "r") as w:
            frames = w.readframes(w.getnframes())
            samples = struct.unpack(f"<{w.getnframes() * w.getnchannels()}h", frames)
        return math.sqrt(sum(s ** 2 for s in samples) / len(samples)) if samples else 0.0

    def _generate_separator(self, amplitude: float) -> Path:
        """Generate separator audio (1s silence + 1s 1000Hz tone + 1s silence).

        Args:
            amplitude: Peak amplitude of the tone in 16-bit sample range.

        Returns:
            Path to generated WAV file (cached by amplitude level).
        """
        # Cache by rounded amplitude to avoid regenerating for tiny differences
        amp_key = int(amplitude)
        path = self.youtube_cache_dir / f"_separator_{amp_key}.wav"
        if path.exists():
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        sr = self.sample_rate
        silence = [0] * int(sr * SEPARATOR_SILENCE_DURATION)
        tone = [
            int(amplitude * math.sin(2 * math.pi * SEPARATOR_FREQ * t / sr))
            for t in range(int(sr * SEPARATOR_TONE_DURATION))
        ]
        samples = silence + tone + silence

        with wave.open(str(path), "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # 16-bit
            w.setframerate(sr)
            w.writeframes(struct.pack(f"<{len(samples)}h", *samples))

        logger.info(f"Generated separator: amplitude={amplitude:.0f} (~{SEPARATOR_AMPLITUDE_RATIO:.0%} of input RMS)")
        return path

    def _get_cache_key(self, url: str, time_interval: str | None) -> str:
        """Generate cache key for YouTube download.

        Args:
            url: YouTube URL
            time_interval: Optional time interval (MM:SS-MM:SS)

        Returns:
            MD5 hash of URL + time_interval
        """
        key_str = f"{url}|{time_interval or ''}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _parse_time_interval(self, interval: str) -> tuple[str, str]:
        """Parse MM:SS-MM:SS into start and end times.

        Args:
            interval: Time interval string (e.g., "00:15-00:30")

        Returns:
            Tuple of (start, end) time strings
        """
        start, end = interval.split("-")
        return start.strip(), end.strip()

    def _time_to_seconds(self, time_str: str) -> float:
        """Convert MM:SS to seconds.

        Args:
            time_str: Time string (e.g., "01:30")

        Returns:
            Time in seconds (e.g., 90.0)
        """
        parts = time_str.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return float(time_str)

    def download_youtube_audio(
        self,
        url: str,
        time_interval: str | None = None,
    ) -> Path | None:
        """Download YouTube audio with caching.

        Downloads audio from YouTube, optionally crops to time interval,
        and caches the result for future use.

        Args:
            url: YouTube URL
            time_interval: Optional MM:SS-MM:SS for cropping

        Returns:
            Path to cached audio file, or None if failed
        """
        cache_key = self._get_cache_key(url, time_interval)
        cache_path = self.youtube_cache_dir / f"{cache_key}.wav"

        # Check cache
        if cache_path.exists():
            logger.debug(f"Cache hit: {url}")
            return cache_path

        logger.info(f"Downloading: {url}")

        # Download with yt-dlp (with retry)
        temp_path = self.youtube_cache_dir / f"{cache_key}_temp.%(ext)s"
        cmd = [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "--postprocessor-args", f"ffmpeg:-ar {self.sample_rate} -ac 1",
            "-o", str(temp_path),
            url,
        ]

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                break  # Success, exit retry loop
            except subprocess.CalledProcessError as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Download attempt {attempt + 1}/{MAX_RETRIES} failed for {url}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"yt-dlp failed after {MAX_RETRIES} attempts for {url}: {e.stderr}")
                    return None

        # Find the downloaded file
        downloaded = list(self.youtube_cache_dir.glob(f"{cache_key}_temp.*"))
        if not downloaded:
            logger.error(f"No file found after download: {url}")
            return None

        downloaded_path = downloaded[0]

        # Crop if time_interval specified
        if time_interval:
            start, end = self._parse_time_interval(time_interval)
            start_sec = self._time_to_seconds(start)
            end_sec = self._time_to_seconds(end)
            duration = end_sec - start_sec

            crop_cmd = [
                "ffmpeg",
                "-i", str(downloaded_path),
                "-ss", str(start_sec),
                "-t", str(duration),
                "-ar", str(self.sample_rate),
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(cache_path),
                "-y",
            ]

            try:
                subprocess.run(crop_cmd, check=True, capture_output=True, text=True)
                downloaded_path.unlink()  # Remove temp file
            except subprocess.CalledProcessError as e:
                logger.error(f"ffmpeg crop failed: {e.stderr}")
                downloaded_path.unlink()
                return None
        else:
            # Just rename to cache path
            downloaded_path.rename(cache_path)

        return cache_path

    def resample_audio(self, input_path: Path, output_path: Path) -> bool:
        """Resample audio to target sample rate.

        Args:
            input_path: Source audio file
            output_path: Destination path

        Returns:
            True if successful, False otherwise
        """
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(output_path),
            "-y",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Resample failed for {input_path}: {e.stderr}")
            return False

    def concatenate_audio(
        self,
        audio_paths: list[Path],
        output_path: Path,
    ) -> bool:
        """Concatenate audio files with a dynamically-leveled beep separator.

        For single audio: just resamples to target sample rate.
        For multiple audios: measures input RMS, generates a proportionally
        quiet separator tone, and concatenates with ffmpeg.

        Args:
            audio_paths: List of audio files to concatenate
            output_path: Output path for combined audio

        Returns:
            True if successful, False otherwise
        """
        if len(audio_paths) == 1:
            return self.resample_audio(audio_paths[0], output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Measure average RMS across input segments, generate separator at 15% of that
        avg_rms = sum(self._measure_rms(p) for p in audio_paths) / len(audio_paths)
        tone_amplitude = max(avg_rms * SEPARATOR_AMPLITUDE_RATIO, 50)  # floor to stay audible
        separator = self._generate_separator(tone_amplitude)

        # Build concat file list
        list_file = output_path.parent / f"{output_path.stem}_list.txt"
        with open(list_file, "w") as f:
            for i, path in enumerate(audio_paths):
                f.write(f"file '{path.absolute()}'\n")
                if i < len(audio_paths) - 1:
                    f.write(f"file '{separator.absolute()}'\n")

        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(output_path),
            "-y",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            list_file.unlink()
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Concatenation failed: {e.stderr}")
            if list_file.exists():
                list_file.unlink()
            return False

    def get_student_audio_path(self, filename: str) -> Path:
        """Get full path to student audio file.

        Args:
            filename: Audio filename (e.g., "c_major_scale.wav")

        Returns:
            Full path to the file
        """
        return self.student_audio_dir / filename
