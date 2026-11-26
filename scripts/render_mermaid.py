"""Render Mermaid diagrams only in the generated website sources."""
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

FENCE = re.compile(r"(?m)^ {0,3}\x60{3,}[ \t]*mermaid[ \t]*$")
IMAGE = re.compile(r"!\[[^\]]*\]\(\./([^)\s]+\.svg)(?:\s+\"[^\"]*\")?\)")

with TemporaryDirectory() as tmp:
    config = Path(tmp) / "puppeteer.json"
    config.write_text('{"args":["--no-sandbox","--disable-setuid-sandbox"]}', encoding="utf-8")
    for source in sorted(Path("deepagents").glob("chapter*.md")):
        text = source.read_text(encoding="utf-8")
        count = len(FENCE.findall(text))
        if not count:
            continue
        target = Path("docs/deepagents") / source.name
        subprocess.run([
            "mmdc", "-i", str(source), "-o", str(target),
            "-p", str(config), "-t", "neutral", "-b", "white",
        ], check=True)
        rendered = target.read_text(encoding="utf-8")
        if FENCE.search(rendered):
            raise RuntimeError(f"{target}: unconverted Mermaid fence")
        images = [name for name in IMAGE.findall(rendered)
                  if name.startswith(target.stem + "-")]
        if len(images) != count:
            raise RuntimeError(f"{target}: expected {count} diagrams, found {len(images)}")
        for name in images:
            path = target.parent / name
            root = ElementTree.parse(path).getroot()
            if root.tag != "{http://www.w3.org/2000/svg}svg":
                raise RuntimeError(f"{path}: not an SVG")
        print(f"{target}: rendered and validated {count} SVG diagrams", flush=True)
