"""Download script for WaveGen dataset.

This script downloads the WaveGenAI/youtube-cc-by-music dataset from HuggingFace
and extracts audio files using yt-dlp. This is a dataset preparation script,
separate from the generic audio loading code in meanflow_audio_codec.datasets.audio.

Following BigVision's tools structure, this can be run as:
    python -m meanflow_audio_codec.tools.download_wavegen [--output-dir OUTPUT_DIR]

Or imported and used programmatically:
    from meanflow_audio_codec.tools import download_wavegen
    download_wavegen.main()
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from yt_dlp import YoutubeDL

MAX_DURATION_SECONDS = 30 * 60  # 30 minutes


def main():
    """Main entry point for downloading WaveGen dataset."""
    parser = argparse.ArgumentParser(
        description="Download WaveGen dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "datasets" / "wavegen",
        help="Directory to save downloaded audio files (default: ~/datasets/wavegen)",
    )
    args = parser.parse_args()

    downloads_dir = args.output_dir
    downloads_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset metadata from HuggingFace...")
    ds = load_dataset("WaveGenAI/youtube-cc-by-music", split="train")

    ytdl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(downloads_dir / "%(id)s.%(ext)s"),
        "download_archive": str(downloads_dir / "archive.txt"),
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "postprocessor_args": {
            "ffmpeg": [
                "-ar", "44100",  # 44.1kHz sample rate
                "-b:a", "128k",  # 128kbps bitrate (CBR)
            ],
        },
    }

    print(f"Downloading audio files to {downloads_dir}...")
    with YoutubeDL(ytdl_opts) as ydl:
        for ex in ds:
            try:
                vid = ex.get("video_id") or ex.get("id")
                url = ex.get("url", f"https://www.youtube.com/watch?v={vid}")

                # Extract info to check duration before downloading
                info = ydl.extract_info(url, download=False)
                duration = info.get("duration", 0)

                if duration > MAX_DURATION_SECONDS:
                    mins = duration // 60
                    secs = duration % 60
                    print(
                        f"Skipping {ex} "
                        f"(duration: {mins}m {secs}s > 30min)"
                    )
                    continue

                mins = duration // 60
                secs = duration % 60
                print(f"Downloading {ex} (duration: {mins}m {secs}s)…")
                ydl.download([url])
            except Exception as er:
                print(f"Error downloading {ex}: {er}")

    print(f"\nDownload complete! Files saved to {downloads_dir}")
    print(f"You can now use this directory with build_audio_pipeline()")


if __name__ == "__main__":
    main()

