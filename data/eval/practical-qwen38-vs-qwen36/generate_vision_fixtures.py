#!/usr/bin/env python3
"""Generate exact-content vision fixtures for the practical 3.8 vs 3.6 suite.

Charts, OCR, invoices, and labeled diagrams are drawn with Pillow so every
pixel-level label is known. Do not replace these with generative images.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    supplemental = Path("/System/Library/Fonts/Supplemental")
    candidates = [
        str(supplemental / ("Arial Bold.ttf" if bold else "Arial.ttf")),
        str(supplemental / "Arial.ttf"),
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def save(img: Image.Image, name: str) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    dest = IMAGES / name
    img.save(dest, format="PNG")
    print(dest)


def count_apples() -> None:
    img = Image.new("RGB", (720, 480), (245, 242, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 300, 680, 440), fill=(196, 164, 132))
    positions = [(90, 220), (190, 200), (290, 230), (390, 195), (490, 225), (150, 280), (430, 275)]
    for x, y in positions:
        draw.ellipse((x, y, x + 70, y + 70), fill=(196, 32, 32), outline=(120, 10, 10), width=2)
        draw.line((x + 35, y + 4, x + 42, y - 18), fill=(40, 110, 40), width=4)
    draw.text((24, 18), "Kitchen table", fill=(40, 40, 40), font=font(28, bold=True))
    save(img, "vl-count-apples.png")


def ocr_code() -> None:
    img = Image.new("RGB", (900, 360), (18, 32, 56))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((60, 70, 840, 290), radius=18, fill=(248, 248, 248))
    draw.text((90, 95), "ACCESS CODE", fill=(80, 80, 80), font=font(28, bold=True))
    draw.text((90, 150), "AXQ-38-M2U", fill=(16, 16, 16), font=font(72, bold=True))
    save(img, "vl-ocr-code.png")


def bar_chart() -> None:
    img = Image.new("RGB", (860, 560), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((40, 20), "Quarterly units sold", fill=(20, 20, 20), font=font(32, bold=True))
    origin = (90, 480)
    max_h = 360
    values = [("Q1", 12), ("Q2", 19), ("Q3", 7), ("Q4", 15)]
    max_v = 20
    colors = [(70, 130, 180), (40, 150, 90), (210, 120, 40), (120, 90, 180)]
    draw.line((origin[0], origin[1], origin[0], origin[1] - max_h), fill=(0, 0, 0), width=2)
    draw.line((origin[0], origin[1], 800, origin[1]), fill=(0, 0, 0), width=2)
    for i, ((label, value), color) in enumerate(zip(values, colors, strict=True)):
        x0 = origin[0] + 60 + i * 170
        h = int(max_h * (value / max_v))
        draw.rectangle((x0, origin[1] - h, x0 + 110, origin[1]), fill=color)
        label_font = font(24, bold=True)
        value_font = font(26, bold=True)
        draw.text((x0 + 28, origin[1] + 12), label, fill=(0, 0, 0), font=label_font)
        draw.text((x0 + 36, origin[1] - h - 36), str(value), fill=(0, 0, 0), font=value_font)
    save(img, "vl-barchart.png")


def spatial() -> None:
    img = Image.new("RGB", (800, 420), (236, 236, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle((80, 120, 250, 290), fill=(40, 90, 210), outline=(10, 30, 90), width=3)
    draw.ellipse((530, 120, 720, 310), fill=(210, 40, 40), outline=(90, 10, 10), width=3)
    draw.text((110, 330), "BLUE CUBE", fill=(20, 20, 20), font=font(22, bold=True))
    draw.text((555, 330), "RED SPHERE", fill=(20, 20, 20), font=font(22, bold=True))
    save(img, "vl-spatial.png")


def invoice() -> None:
    img = Image.new("RGB", (800, 640), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 800, 80), fill=(20, 50, 90))
    draw.text((32, 22), "INVOICE", fill=(255, 255, 255), font=font(36, bold=True))
    rows = [
        ("Invoice number", "4821"),
        ("Customer", "Northwind Labs"),
        ("Item A", "$80.00"),
        ("Item B", "$42.50"),
        ("Tax", "$14.00"),
        ("TOTAL", "$136.50"),
    ]
    y = 120
    for label, value in rows:
        bold = label == "TOTAL"
        draw.text((48, y), label, fill=(20, 20, 20), font=font(28, bold=bold))
        draw.text((520, y), value, fill=(20, 20, 20), font=font(28, bold=bold))
        y += 70
    draw.line((48, 470, 752, 470), fill=(20, 20, 20), width=2)
    save(img, "vl-invoice.png")


def clock() -> None:
    img = Image.new("RGB", (520, 520), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.ellipse((40, 40, 480, 480), outline=(20, 20, 20), width=8, fill=(255, 255, 255))
    cx, cy, r = 260, 260, 200
    for hour in range(12):
        ang = math.radians(hour * 30 - 90)
        x = cx + int(math.cos(ang) * (r - 18))
        y = cy + int(math.sin(ang) * (r - 18))
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(20, 20, 20))

    # 3:15 — hour at 97.5°, minute at 90° from 12 o'clock? 3:00 hour at +90deg from 12.
    # minute 15 -> 90 degrees from 12; hour 3 + 15/60 -> 97.5 deg.
    hour_ang = math.radians(97.5 - 90)
    min_ang = math.radians(90 - 90)
    hour_end = (
        cx + int(math.cos(hour_ang) * 110),
        cy + int(math.sin(hour_ang) * 110),
    )
    minute_end = (
        cx + int(math.cos(min_ang) * 160),
        cy + int(math.sin(min_ang) * 160),
    )
    draw.line((cx, cy, *hour_end), fill=(20, 20, 20), width=10)
    draw.line((cx, cy, *minute_end), fill=(180, 30, 30), width=6)
    draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=(20, 20, 20))
    save(img, "vl-clock.png")


def triangle() -> None:
    img = Image.new("RGB", (640, 520), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    pts = [(80, 430), (80 + 3 * 80, 430), (80, 430 - 4 * 80)]
    draw.polygon(pts, outline=(20, 20, 20), fill=(230, 240, 255))
    draw.line([*pts, pts[0]], fill=(20, 20, 20), width=5)
    draw.rectangle((80, 400, 110, 430), outline=(20, 20, 20), width=3)
    draw.text((180, 445), "3", fill=(0, 0, 0), font=font(36, bold=True))
    draw.text((20, 250), "4", fill=(0, 0, 0), font=font(36, bold=True))
    draw.text((250, 220), "?", fill=(180, 20, 20), font=font(40, bold=True))
    draw.text(
        (40, 20),
        "Right triangle (right angle at the square)",
        fill=(20, 20, 20),
        font=font(22, bold=True),
    )
    save(img, "vl-triangle.png")


def traffic() -> None:
    img = Image.new("RGB", (280, 640), (40, 40, 40))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((70, 40, 210, 600), radius=30, fill=(20, 20, 20))
    draw.ellipse((95, 70, 185, 160), fill=(80, 20, 20))
    draw.ellipse((95, 250, 185, 340), fill=(80, 70, 20))
    draw.ellipse((95, 430, 185, 520), fill=(30, 200, 60))
    save(img, "vl-traffic.png")


def button() -> None:
    img = Image.new("RGB", (720, 280), (245, 247, 250))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((160, 80, 560, 200), radius=16, fill=(20, 110, 80))
    draw.text((250, 115), "CONFIRM", fill=(255, 255, 255), font=font(42, bold=True))
    save(img, "vl-button.png")


def grid_star() -> None:
    img = Image.new("RGB", (540, 540), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(4):
        draw.line((30, 30 + i * 160, 510, 30 + i * 160), fill=(0, 0, 0), width=3)
        draw.line((30 + i * 160, 30, 30 + i * 160, 510), fill=(0, 0, 0), width=3)
    # star in center cell
    cx, cy = 270, 270
    pts = []
    for i in range(10):
        ang = math.radians(-90 + i * 36)
        rad = 55 if i % 2 == 0 else 24
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    draw.polygon(pts, fill=(220, 170, 20), outline=(120, 80, 0))
    save(img, "vl-grid-star.png")


def pie() -> None:
    img = Image.new("RGB", (640, 520), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    box = (80, 40, 480, 440)
    draw.pieslice(box, start=0, end=180, fill=(50, 110, 200))
    draw.pieslice(box, start=180, end=270, fill=(230, 130, 30))
    draw.pieslice(box, start=270, end=360, fill=(50, 170, 80))
    draw.ellipse(box, outline=(20, 20, 20), width=3)
    draw.rectangle((500, 80, 530, 110), fill=(50, 110, 200))
    draw.text((540, 80), "Blue 50%", fill=(0, 0, 0), font=font(20, bold=True))
    draw.rectangle((500, 140, 530, 170), fill=(230, 130, 30))
    draw.text((540, 140), "Orange 25%", fill=(0, 0, 0), font=font(20, bold=True))
    draw.rectangle((500, 200, 530, 230), fill=(50, 170, 80))
    draw.text((540, 200), "Green 25%", fill=(0, 0, 0), font=font(20, bold=True))
    draw.text((40, 470), "Market share", fill=(0, 0, 0), font=font(24, bold=True))
    save(img, "vl-pie.png")


def letter_date() -> None:
    img = Image.new("RGB", (800, 640), (252, 248, 236))
    draw = ImageDraw.Draw(img)
    draw.text((80, 60), "Northwind Labs", fill=(20, 20, 20), font=font(28, bold=True))
    draw.text((80, 120), "14 August 2026", fill=(20, 20, 20), font=font(26))
    draw.text((80, 200), "Dear colleague,", fill=(20, 20, 20), font=font(26))
    draw.text((80, 260), "Please confirm receipt of shipment", fill=(20, 20, 20), font=font(26))
    draw.text((80, 310), "batch NW-771.", fill=(20, 20, 20), font=font(26))
    draw.text((80, 400), "Regards,", fill=(20, 20, 20), font=font(26))
    draw.text((80, 450), "Operations", fill=(20, 20, 20), font=font(26))
    save(img, "vl-letter-date.png")


def shapes() -> None:
    img = Image.new("RGB", (800, 420), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    yellow = [(70, 140), (200, 140), (330, 140)]
    for x, y in yellow:
        draw.ellipse((x, y, x + 90, y + 90), fill=(240, 210, 40), outline=(140, 110, 10), width=3)
    purple = [(500, 90), (650, 220)]
    for x, y in purple:
        draw.rectangle((x, y, x + 90, y + 90), fill=(130, 70, 180), outline=(60, 20, 90), width=3)
    save(img, "vl-shapes.png")


def path() -> None:
    img = Image.new("RGB", (720, 480), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # S --right--> * --down--> * --right--> E   => 2 right turns if facing along path?
    # Start facing east, go east, turn right (south), turn right (west) would go left of E.
    # Simpler: a path with arrows: start S at left, go right, turn down, turn right to E.
    # From S east, then south (right turn), then east (left turn) = 1 right + 1 left.
    # User question will ask how many right-angle turns, not direction.
    draw.line((80, 120, 360, 120, 360, 340, 640, 340), fill=(30, 80, 180), width=10)
    draw.polygon([(360, 120), (340, 100), (380, 100)], fill=(30, 80, 180))  # decorative
    draw.ellipse((60, 100, 100, 140), fill=(20, 150, 70))
    draw.ellipse((620, 320, 660, 360), fill=(200, 40, 40))
    draw.text((50, 150), "S", fill=(0, 0, 0), font=font(28, bold=True))
    draw.text((640, 370), "E", fill=(0, 0, 0), font=font(28, bold=True))
    draw.text((40, 20), "Path from S to E", fill=(0, 0, 0), font=font(24, bold=True))
    save(img, "vl-path.png")


def color_block() -> None:
    img = Image.new("RGB", (640, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((80, 80, 280, 320), fill=(0, 128, 128))
    draw.rectangle((360, 80, 560, 320), fill=(255, 165, 0))
    draw.text((120, 340), "LEFT", fill=(0, 0, 0), font=font(22, bold=True))
    draw.text((410, 340), "RIGHT", fill=(0, 0, 0), font=font(22, bold=True))
    save(img, "vl-colors.png")


def main() -> None:
    count_apples()
    ocr_code()
    bar_chart()
    spatial()
    invoice()
    clock()
    triangle()
    traffic()
    button()
    grid_star()
    pie()
    letter_date()
    shapes()
    path()
    color_block()


if __name__ == "__main__":
    main()
