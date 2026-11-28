"""Collect freely licensed reference photos (Pixabay/Pexels/Unsplash) for demos.

Bing image search is used only as a discovery index; downloads are restricted to
the free-licence CDNs and every file records its source URL in manifest.json.
These photos are reference material for trying the annotation workflow, not
training data for a specific test site.
"""

from __future__ import annotations

import html
import io
import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

ALLOWED_HOSTS = ("images.pexels.com", "cdn.pixabay.com", "images.unsplash.com")
QUERIES = {
    "gully": "pixabay erosion gully",
    "soil": "pixabay soil erosion",
    "collapse": "pixabay landslide",
    "protection": "pixabay erosion control mat",
    "rain": "pixabay rain soil",
    "slope": "pixabay bare slope",
}
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")
CTX = ssl.create_default_context()


def search_bing(query: str, count: int = 35):
    url = ("https://cn.bing.com/images/async?q=" + urllib.parse.quote(query)
           + f"&first=1&count={count}&mkt=en-US")
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Referer": "https://cn.bing.com/images/search?q=" + urllib.parse.quote(query),
        "Accept-Language": "en-US,en;q=0.9",
    })
    with urllib.request.urlopen(request, timeout=30, context=CTX) as response:
        page = html.unescape(response.read().decode("utf-8", errors="replace"))
    found = []
    for match in re.finditer(r'"murl":"(https?://[^"]+)"', page):
        media = match.group(1)
        host = urllib.parse.urlparse(media).netloc.lower()
        if any(host.endswith(allowed) for allowed in ALLOWED_HOSTS):
            found.append(media)
    return found


def download(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Referer": "https://www.pixabay.com/"})
    with urllib.request.urlopen(request, timeout=60, context=CTX) as response:
        data = response.read()
    if len(data) < 20_000:
        raise ValueError(f"too small ({len(data)} bytes)")
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        if min(width, height) < 600:
            raise ValueError(f"resolution {width}x{height} too small")
    return data, (width, height)


def source_page(media: str) -> str:
    match = re.search(r"/photo/(\d{4})/(\d{2})/(\d{2})/(\d{2})/(\d{2})/([a-z0-9-]+?)-(\d+)_", media)
    if match:
        return f"https://pixabay.com/photos/{match.group(6)}-{match.group(7)}/"
    return media


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "web_photos"
    out.mkdir(parents=True, exist_ok=True)
    manifest, seen = [], set()
    for theme, query in QUERIES.items():
        try:
            candidates = search_bing(query)
        except Exception as error:  # noqa: BLE001
            print(f"{theme}: search failed: {error}")
            continue
        print(f"{theme}: {len(candidates)} candidates")
        taken = 0
        for media in candidates:
            if taken >= 4:
                break
            if media in seen:
                continue
            seen.add(media)
            suffix = ".png" if ".png" in media.lower() else ".jpg"
            target = out / f"{theme}_{taken + 1:02d}{suffix}"
            try:
                data, size = download(media)
            except Exception as error:  # noqa: BLE001
                print(f"  skip {media[:70]} ({error})")
                continue
            target.write_bytes(data)
            manifest.append({
                "file": target.name,
                "theme": theme,
                "query": query,
                "media": media,
                "source": source_page(media),
                "size": list(size),
                "license": "Pixabay Content License（免费商用，建议署名）",
            })
            taken += 1
            print(f"  saved {target.name} {size[0]}x{size[1]} {len(data)/1024:.0f} KB")
            time.sleep(0.3)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: {len(manifest)} photos -> {out}")


if __name__ == "__main__":
    main()
