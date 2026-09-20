#!/usr/bin/env python3

from pathlib import Path
import sys

LINES_PER_CHUNK = 1000

if len(sys.argv) != 2:
    print(f"Usage: {Path(sys.argv[0]).name} <filename>")
    sys.exit(1)

source = Path(sys.argv[1])

if not source.is_file():
    print(f"Error: file not found: {source}")
    sys.exit(1)

with source.open("r", encoding="utf-8") as f:
    lines = f.readlines()

total_chunks = (len(lines) + LINES_PER_CHUNK - 1) // LINES_PER_CHUNK

for chunk_num in range(total_chunks):
    start = chunk_num * LINES_PER_CHUNK
    end = start + LINES_PER_CHUNK

    output_name = (
        f"{source.stem}_part_{chunk_num + 1:03d}.txt"
    )
    output_path = source.with_name(output_name)

    with output_path.open("w", encoding="utf-8") as out:
        out.writelines(lines[start:end])

    print(
        f"Created {output_name} "
        f"({start + 1}-{min(end, len(lines))})"
    )

print(
    f"\nDone. Split {len(lines)} lines into "
    f"{total_chunks} files."
)