"""Generate a deterministic SHA-256 manifest for the Git-tracked release tree."""
from __future__ import annotations
import argparse, hashlib, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "RELEASE_MANIFEST.sha256"

def tracked_files():
    proc = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], check=True, capture_output=True)
    rels = [Path(p.decode("utf-8")) for p in proc.stdout.split(b"\0") if p]
    return sorted(p for p in rels if p.as_posix() != MANIFEST.name)

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()

def build_manifest():
    return "\n".join(f"{sha256(ROOT/p)}  ./{p.as_posix()}" for p in tracked_files())+"\n"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--write",action="store_true"); a=ap.parse_args(); c=build_manifest()
    if a.write: MANIFEST.write_text(c,encoding="utf-8")
    else: print(c,end="")
if __name__=="__main__": main()
