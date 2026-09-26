"""Build <team>_submission.zip with the layout required by the challenge README."""
from __future__ import annotations

import os
import sys
import zipfile

from io_utils import OUT, ROOT

TEAM = "YAAD"


def main(team: str = TEAM) -> str:
    path = os.path.join(ROOT, f"{team}_submission.zip")
    code_dir = os.path.join(ROOT, "code", "business_entity_resolution")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(os.path.join(OUT, name), f"output/{name}")
        for dirpath, _, files in os.walk(code_dir):
            if "__pycache__" in dirpath:
                continue
            for fn in files:
                if fn.endswith((".pyc", ".DS_Store")):
                    continue
                full = os.path.join(dirpath, fn)
                z.write(full, os.path.join("code", "business_entity_resolution", os.path.relpath(full, code_dir)))
        z.write(os.path.join(ROOT, "Documentation_template.md"), "Documentation_template.md")
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.0f} MB")
    return path


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
