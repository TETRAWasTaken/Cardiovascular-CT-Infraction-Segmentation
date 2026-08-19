"""Extract the split ZIP archives that make up the ImageCAS dataset.

Each ImageCAS range consists of one final ``.zip`` volume and one or more
preceding ``.zNN`` volumes.  All volumes must remain in the same directory;
the final ``.zip`` file is passed to 7-Zip, or merged with the existing
``zip`` utility when 7-Zip is unavailable.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


VOLUME_PATTERN = re.compile(r"^(?P<stem>.+)\.z(?P<number>\d{2})$", re.IGNORECASE)


def find_archives(data_dir: Path) -> list[Path]:
	"""Return final ZIP volumes whose split parts are present and complete."""
	archives = sorted(data_dir.glob("*.zip"))
	if not archives:
		raise FileNotFoundError(f"No .zip archives found in {data_dir}")

	for archive in archives:
		parts = sorted(
			(
				path
				for path in data_dir.glob(f"{archive.stem}.z[0-9][0-9]")
				if VOLUME_PATTERN.match(path.name)
			),
			key=lambda path: int(VOLUME_PATTERN.match(path.name).group("number")),
		)
		if not parts:
			raise FileNotFoundError(f"Missing split volumes for {archive.name}")

		expected = list(range(1, len(parts) + 1))
		actual = [int(VOLUME_PATTERN.match(path.name).group("number")) for path in parts]
		if actual != expected:
			raise FileNotFoundError(
				f"Split volumes for {archive.name} must be consecutive from .z01; "
				f"found {actual}"
			)

	return archives


def extract_archive(
	archive: Path, output_dir: Path, extractor: str, dry_run: bool = False
) -> None:
	"""Validate and extract one split archive with a multi-volume extractor."""
	print(f"{'Would extract' if dry_run else 'Extracting'} {archive.name} -> {output_dir}")
	command = [extractor, "t", str(archive)]
	if not dry_run:
		output_dir.mkdir(parents=True, exist_ok=True)
		command = [extractor, "x", str(archive), f"-o{output_dir}", "-y"]
	result = subprocess.run(command, capture_output=True, text=True, check=False)
	if result.returncode != 0:
		details = result.stderr.strip() or result.stdout.strip()
		raise RuntimeError(f"{archive.name} could not be processed: {details}")


def extract_with_zip(archive: Path, output_dir: Path, dry_run: bool = False) -> None:
	"""Merge a split archive with the existing zip utility, then extract it."""
	merged_archive = archive.with_name(f"{archive.stem}-merged.zip")
	print(f"{'Would merge' if dry_run else 'Merging'} {archive.name} -> {merged_archive.name}")
	if not dry_run:
		result = subprocess.run(
			["zip", "-s", "0", str(archive), "--out", str(merged_archive)],
			capture_output=True,
			text=True,
			check=False,
		)
		if result.returncode != 0:
			details = result.stderr.strip() or result.stdout.strip()
			raise RuntimeError(f"{archive.name} could not be merged: {details}")

	try:
		if not dry_run:
			output_dir.mkdir(parents=True, exist_ok=True)
			with zipfile.ZipFile(merged_archive) as zip_file:
				zip_file.extractall(output_dir)
	finally:
		if not dry_run:
			merged_archive.unlink(missing_ok=True)


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"data_dir",
		nargs="?",
		type=Path,
		default=Path(__file__).resolve().parents[1] / "Data",
		help="directory containing the .zip and .zNN files (default: ../Data)",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		help="directory to extract into (default: <data_dir>/extracted)",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="validate archives and show actions without extracting files",
	)
	args = parser.parse_args()
	data_dir = args.data_dir.expanduser().resolve()
	output_dir = (args.output or data_dir / "extracted").expanduser().resolve()

	try:
		archives = find_archives(data_dir)
		extractor = shutil.which("7zz") or shutil.which("7z")
		zip_utility = shutil.which("zip")
		if extractor is None and zip_utility is None:
			raise RuntimeError(
				"Neither 7-Zip nor the zip utility is available. "
				"Use an existing archive tool or install one."
			)
		for archive in archives:
			if extractor is not None:
				extract_archive(archive, output_dir, extractor, dry_run=args.dry_run)
			else:
				extract_with_zip(archive, output_dir, dry_run=args.dry_run)
	except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
		print(f"Error: {error}", file=sys.stderr)
		return 1

	print(f"{'Validation complete' if args.dry_run else 'Extraction complete'}: {len(archives)} archives")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
