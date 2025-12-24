"""Utility script to migrate config files from v1.0 (flat) to v2.0 (hierarchical) format."""

import json
import sys
from pathlib import Path
from typing import Any

from meanflow_audio_codec.configs.config import migrate_config_v1_to_v2, load_config_from_json


def migrate_config_file(input_path: Path, output_path: Path | None = None) -> None:
    """Migrate a single config file from v1.0 to v2.0 format.
    
    Args:
        input_path: Path to input config file (v1.0 format)
        output_path: Path to output config file (v2.0 format). If None, overwrites input.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Config file not found: {input_path}")
    
    # Load config (auto-migration happens in load_config_from_json)
    config = load_config_from_json(input_path)
    
    # Determine output path
    if output_path is None:
        output_path = input_path
    
    # Save migrated config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, sort_keys=True)
    
    print(f"Migrated: {input_path} -> {output_path}")


def migrate_config_directory(
    input_dir: Path,
    output_dir: Path | None = None,
    pattern: str = "*.json",
) -> None:
    """Migrate all config files in a directory.
    
    Args:
        input_dir: Directory containing config files
        output_dir: Output directory. If None, overwrites files in place.
        pattern: Glob pattern for config files (default: "*.json")
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    config_files = list(input_dir.glob(pattern))
    if not config_files:
        print(f"No config files found in {input_dir} matching pattern {pattern}")
        return
    
    print(f"Found {len(config_files)} config file(s) to migrate")
    
    for config_file in config_files:
        if output_dir is None:
            output_path = config_file
        else:
            output_path = output_dir / config_file.name
            output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            migrate_config_file(config_file, output_path)
        except Exception as e:
            print(f"Error migrating {config_file}: {e}", file=sys.stderr)


def main():
    """Main entry point for migration script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate config files from v1.0 (flat) to v2.0 (hierarchical) format"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input config file or directory",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output config file or directory (default: overwrite input)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern for config files when input is a directory (default: *.json)",
    )
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    if input_path.is_file():
        migrate_config_file(input_path, output_path)
    elif input_path.is_dir():
        migrate_config_directory(input_path, output_path, args.pattern)
    else:
        print(f"Error: {input_path} is not a file or directory", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


