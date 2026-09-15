"""
Access layer for a DIALux evo project archive.

An .evo file is a ZIP container. This module lets the rest of the tool work
against either the raw .evo file or an already-extracted folder, using the
same API.
"""

from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

# ---------------------------------------------------------------------------
# File signature detection: many payloads inside an .evo have stripped or fake
# extensions (.tmp, or none at all). We sniff the real type from magic bytes.
# ---------------------------------------------------------------------------

MAGIC_TABLE = [
    (b"\x89PNG\r\n\x1a\n", "png", "PNG image"),
    (b"\xff\xd8\xff", "jpg", "JPEG image"),
    (b"GIF87a", "gif", "GIF image"),
    (b"GIF89a", "gif", "GIF image"),
    (b"BM", "bmp", "BMP image"),
    (b"II*\x00", "tif", "TIFF image (little endian)"),
    (b"MM\x00*", "tif", "TIFF image (big endian)"),
    (b"PK\x03\x04", "zip", "ZIP archive (nested)"),
    (b"PK\x05\x06", "zip", "ZIP archive (empty)"),
    (b"%PDF", "pdf", "PDF document"),
    (b"\x1f\x8b", "gz", "gzip stream"),
    (b"RIFF", "riff", "RIFF container"),
    (b"<?xml", "xml", "XML text"),
    (b"\xef\xbb\xbf<?xml", "xml", "XML text (UTF-8 BOM)"),
    (b"{", "json", "JSON text (probable)"),
    (b"\x00\x01\x00\x00", "bin", "binary, possible .NET/struct header"),
]


def sniff(data: bytes) -> tuple[str, str]:
    """Return (extension, human description) guessed from the leading bytes."""
    head = data[:16]
    for magic, ext, desc in MAGIC_TABLE:
        if head.startswith(magic):
            return ext, desc
    # Text heuristic: mostly printable ASCII in the first block
    sample = data[:512]
    if sample:
        printable = sum(1 for b in sample if 9 <= b <= 13 or 32 <= b <= 126)
        if printable / len(sample) > 0.92:
            return "txt", "plain text"
    return "bin", "unrecognised binary"


@dataclass
class Entry:
    """One file inside the archive."""

    path: str                 # normalised, forward-slash, relative to archive root
    size: int
    _read: object = field(repr=False, default=None)

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def folder(self) -> str:
        return self.path.rsplit("/", 1)[0] if "/" in self.path else ""

    def read(self) -> bytes:
        return self._read()

    def sniff(self) -> tuple[str, str]:
        return sniff(self.read()[:1024])


class EvoArchive:
    """
    Uniform read access to a DIALux evo project.

    >>> arc = EvoArchive.open("Project10.evo")          # the raw file
    >>> arc = EvoArchive.open("Project10.evo/")         # an extracted folder
    >>> arc.get("Project/ProjectData/ProjectData.xml").read()
    """

    def __init__(self, root: Path, entries: Dict[str, Entry], kind: str):
        self.root = root
        self._entries = entries
        self.kind = kind  # "zip" or "dir"

    # -- construction --------------------------------------------------------

    @classmethod
    def open(cls, path: str | os.PathLike) -> "EvoArchive":
        p = Path(path)
        if not p.exists():
            cwd = Path.cwd()
            siblings = sorted(x.name for x in cwd.iterdir())[:25]
            hint = "\n".join(f"    {s}" for s in siblings) or "    (empty)"
            raise FileNotFoundError(
                f"'{path}' does not exist.\n"
                f"  Current directory: {cwd}\n"
                f"  Files/folders here:\n{hint}\n"
                f"  Pass the full path to Project10.evo, e.g.:\n"
                f'    python -m evostudy inspect "C:\\full\\path\\to\\Project10.evo"'
            )

        if p.is_dir():
            return cls._from_dir(p)

        # A .evo file is a zip; also accept a plain .zip
        if zipfile.is_zipfile(p):
            return cls._from_zip(p)

        raise ValueError(
            f"{p} is neither a folder nor a ZIP container. "
            "If this is a .evo file, make sure it is not corrupted."
        )

    @classmethod
    def _from_dir(cls, root: Path) -> "EvoArchive":
        entries: Dict[str, Entry] = {}
        for f in root.rglob("*"):
            if f.is_file():
                rel = f.relative_to(root).as_posix()
                entries[rel] = Entry(
                    rel, f.stat().st_size, (lambda fp=f: fp.read_bytes())
                )
        # Tolerate the common case of extracting into a wrapper folder
        if not any(k.startswith("Project/") for k in entries):
            subs = [d for d in root.iterdir() if d.is_dir()]
            for s in subs:
                if (s / "Project").is_dir():
                    return cls._from_dir(s)
        return cls(root, entries, "dir")

    @classmethod
    def _from_zip(cls, path: Path) -> "EvoArchive":
        zf = zipfile.ZipFile(path)
        entries: Dict[str, Entry] = {}
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = info.filename.replace("\\", "/")
            entries[rel] = Entry(
                rel, info.file_size, (lambda n=info.filename: zf.read(n))
            )
        arc = cls(path, entries, "zip")
        arc._zf = zf
        return arc

    # -- lookup --------------------------------------------------------------

    def __iter__(self) -> Iterator[Entry]:
        for k in sorted(self._entries):
            yield self._entries[k]

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, path: str) -> Optional[Entry]:
        """Exact path lookup (case-insensitive fallback)."""
        if path in self._entries:
            return self._entries[path]
        low = path.lower()
        for k, v in self._entries.items():
            if k.lower() == low:
                return v
        return None

    def find(self, *, name: str = None, folder: str = None,
             suffix: str = None) -> List[Entry]:
        """Filter entries by filename / folder / suffix, all case-insensitive."""
        out = []
        for e in self:
            if name and name.lower() not in e.name.lower():
                continue
            if folder and folder.lower() not in e.folder.lower():
                continue
            if suffix and not e.name.lower().endswith(suffix.lower()):
                continue
            out.append(e)
        return out

    # -- convenience accessors for the known evo layout ----------------------

    @property
    def archive_info(self) -> Optional[Entry]:
        return self.get("ArchiveInfo/ArchiveInfo.xml")

    @property
    def thumbnail(self) -> Optional[Entry]:
        return self.get("ArchiveInfo/Thumbnail.png")

    @property
    def project_xml(self) -> Optional[Entry]:
        return self.get("Project/ProjectData/ProjectData.xml")

    @property
    def project_dat(self) -> Optional[Entry]:
        return self.get("Project/ProjectData/ProjectData.dat")

    @property
    def output_settings(self) -> Optional[Entry]:
        return self.get("Project/Output/OutputSettings.xml")

    @property
    def scenegraph(self) -> Optional[Entry]:
        return self.get("Project/ScenegraphScene")

    def result_groups(self) -> Dict[str, List[Entry]]:
        """Map 'Group_xx' -> its .rsl files."""
        groups: Dict[str, List[Entry]] = {}
        for e in self.find(folder="Project/Results", suffix=".rsl"):
            parts = e.folder.split("/")
            if len(parts) >= 3:
                groups.setdefault(parts[2], []).append(e)
        return groups

    def m3d_models(self) -> List[Entry]:
        return self.find(suffix=".m3d")

    def product_images(self) -> List[Entry]:
        return self.find(folder="Project/ProductImage")

    def view_thumbnails(self) -> List[Entry]:
        return self.find(folder="Project/ViewThumbnails")

    # -- export --------------------------------------------------------------

    def export_images(self, dest: str | os.PathLike) -> List[Path]:
        """
        Write every image-like payload out with its real extension restored.
        Covers Thumbnail.png, ProductImage/* and ViewThumbnails/*.tmp.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        written = []
        candidates = (
            self.product_images()
            + self.view_thumbnails()
            + self.find(folder="Project/Pool")
            + ([self.thumbnail] if self.thumbnail else [])
        )
        for e in candidates:
            if e is None:
                continue
            data = e.read()
            ext, _ = sniff(data)
            if ext not in {"png", "jpg", "gif", "bmp", "tif"}:
                continue
            stem = e.path.replace("/", "__").rsplit(".", 1)[0]
            out = dest / f"{stem}.{ext}"
            out.write_bytes(data)
            written.append(out)
        return written
