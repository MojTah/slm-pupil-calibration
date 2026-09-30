"""Check every frozen numerical input against the public SHA256 manifest."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
records = json.loads((root/'DATA_MANIFEST.json').read_text())
for record in records:
    path = (root/record['path']).resolve()
    if not path.is_relative_to(root/'local-results'):
        raise ValueError('Manifest path outside numerical inputs')
    if path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Changed input: '+record['path'])
print(f'PASS: {len(records)} frozen input hashes')
