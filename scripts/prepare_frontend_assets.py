"""Vendor the existing reference artwork and OFL fonts for an offline-friendly UI."""
from pathlib import Path
import re
import urllib.request
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter

root = Path(__file__).resolve().parents[1]
assets = root / 'frontend' / 'src' / 'assets'
assets.mkdir(parents=True, exist_ok=True)
source = root / 'assets' / 'danish-hoved-fra-statuen-af-david.jpg'
if not source.exists():
    raise FileNotFoundError('Download the credited public-domain David photograph with the design asset script first.')
portrait = Image.open(source).convert('RGB')
pixels = np.array(portrait)
background = (pixels.max(axis=2) < 22).astype('uint8')
_, labels = cv2.connectedComponents(background)
alpha = np.where(labels == labels[0, 0], 0, 255).astype('uint8')
portrait.putalpha(Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(0.5)))
portrait = ImageOps.mirror(portrait)
portrait.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
portrait.save(assets / 'sculpture.webp', quality=90)
request = urllib.request.Request(
    'https://fonts.googleapis.com/css2?family=Manrope:wght@400..800&family=Outfit:wght@400..800&display=swap',
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'})
css = urllib.request.urlopen(request, timeout=30).read().decode()
rules = []
for family in ('Manrope', 'Outfit'):
    blocks = re.findall(r'@font-face\s*\{[^}]+\}', css)
    matches = [block for block in blocks if f"'{family}'" in block]
    block = matches[-1]
    font_url = re.search(r'url\(([^)]+)\)', block).group(1)
    extension = 'woff2' if 'woff2' in block else 'ttf'
    filename = f'{family.lower()}-latin.{extension}'
    (assets / filename).write_bytes(urllib.request.urlopen(font_url, timeout=30).read())
    rules.append(block.replace(font_url, './' + filename))
(assets / 'fonts.css').write_text('\n'.join(rules), encoding='utf-8')
for source_name, asset_name in (('trump.jpg', 'real-sample.jpg'), ('fake_face_sample.png', 'fake-sample.png')):
    (assets / asset_name).write_bytes((root / 'frontend' / 'samples' / source_name).read_bytes())
print('Prepared credited sculpture cutout, variable fonts, and user-provided sample portraits.')
