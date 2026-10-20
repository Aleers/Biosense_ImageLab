"""Publish related artifacts with rollback if a file replacement fails."""
import os
from pathlib import Path
from uuid import uuid4

class ArtifactPublisher:
    def publish(self, pairs):
        pairs = [(Path(source), Path(target)) for source, target in pairs]
        backups = {}
        published = []
        try:
            for source, target in pairs:
                if not source.is_file():
                    raise ValueError(f'Missing staged artifact: {source.name}')
                if target.exists():
                    backup = target.with_name(f'.{target.name}.{uuid4().hex}.backup')
                    os.replace(target, backup)
                    backups[target] = backup
                os.replace(source, target)
                published.append(target)
        except Exception:
            for target in reversed(published):
                if target.exists():
                    target.unlink()
            for target, backup in backups.items():
                os.replace(backup, target)
            raise
        else:
            for backup in backups.values():
                backup.unlink()
