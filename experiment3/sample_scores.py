#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse, random, csv, math, base64, io, hashlib
import datetime
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image
from openai import OpenAI

parser = argparse.ArgumentParser(
    description="Map Evaluator's color preference as utility scores (0-10 scale) (with palette image)."
)
parser.add_argument("--evaluator_model", default="gpt-4o")
parser.add_argument("--trials", type=int, default=20)
parser.add_argument("--items_per_trial", type=int, default=250)
parser.add_argument("--evaluator_temp", type=float, default=0.4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--outdir", default=None)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

random.seed(args.seed)

EVALUATOR_INFO = """\
I'm looking for the color of a fresh cucumber.
"""

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_with_image(model: str, system_msg: str, user_msg: str, image_b64: str | None,
                    temperature: float = 0.25, debug_tag: str = "") -> str:
    user_parts = [{"type": "input_text", "text": user_msg}]
    if image_b64:
        user_parts.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{image_b64}"
        })

    payload = [
        {"role": "system", "content": [{"type": "input_text", "text": system_msg}]},
        {"role": "user",   "content": user_parts},
    ]

    if args.debug and image_b64:
        raw = base64.b64decode(image_b64)
        sha = hashlib.sha256(raw).hexdigest()[:16]
        try:
            w, h = Image.open(io.BytesIO(raw)).size
        except Exception:
            w, h = -1, -1
        print(f"[DEBUG {debug_tag}] attached PNG: {len(raw)} bytes, {w}x{h}, sha256[:16]={sha}")

    resp = client.responses.create(model=model, temperature=temperature, input=payload)
    return resp.output_text

# LCh(ab) -> sRGB (for image and visualization)
Xn, Yn, Zn = 95.047, 100.000, 108.883
M_XYZ_to_sRGB = (
    ( 3.2406, -1.5372, -0.4986),
    (-0.9689,  1.8758,  0.0415),
    ( 0.0557, -0.2040,  1.0570),
)

def _lch_to_lab(L: float, C: float, h_deg: float):
    h = math.radians(h_deg % 360.0)
    return (L, C*math.cos(h), C*math.sin(h))

def _lab_to_xyz(L: float, a: float, b: float):
    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)
    def finv(t: float): return t**3 if t**3 > 0.008856 else (t - 16.0/116.0) / 7.787
    return (finv(fx) * Xn, finv(fy) * Yn, finv(fz) * Zn)

def _xyz_to_srgb01(X: float, Y: float, Z: float):
    x, y, z = X/100.0, Y/100.0, Z/100.0
    r = M_XYZ_to_sRGB[0][0]*x + M_XYZ_to_sRGB[0][1]*y + M_XYZ_to_sRGB[0][2]*z
    g = M_XYZ_to_sRGB[1][0]*x + M_XYZ_to_sRGB[1][1]*y + M_XYZ_to_sRGB[1][2]*z
    b = M_XYZ_to_sRGB[2][0]*x + M_XYZ_to_sRGB[2][1]*y + M_XYZ_to_sRGB[2][2]*z
    def compand(u: float): return 12.92*u if u <= 0.0031308 else 1.055*(max(u,0.0)**(1/2.4)) - 0.055
    R,G,B = compand(r), compand(g), compand(b)
    return (min(max(R,0.0),1.0), min(max(G,0.0),1.0), min(max(B,0.0),1.0))

def lch_to_rgb01(L: float, h_turn: float, C_norm: float):
    h_deg = (h_turn % 1.0) * 360.0
    C = max(0.0, min(1.0, C_norm)) * 100.0
    return _xyz_to_srgb01(*_lab_to_xyz(*_lch_to_lab(L, C, h_deg)))

def create_color_bar_image(lch_list: List[dict], width: int | None = None, height: int = 50) -> str:
    """
    lch_list: [{"L":70.0, "h":..., "c":...}, ...]
    returns base64 PNG
    """
    n = max(1, len(lch_list))
    img_w = width if width else max(800, 12*n)  
    seg_w = max(1, img_w // n)
    img = Image.new('RGB', (img_w, height))
    for i, c in enumerate(lch_list):
        rgb01 = lch_to_rgb01(c["L"], c["h"], c["c"])
        rgb255 = tuple(int(round(x*255)) for x in rgb01)
        x0 = i * seg_w
        x1 = img_w if i == n-1 else x0 + seg_w
        for y in range(height):
            for x in range(x0, x1):
                img.putpixel((x, y), rgb255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def save_png(b64_png: str, path: str):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_png))

# Evaluator system prompt
SYS_EVALUATOR_SCORER = f"""
You are an Evaluator tasked with assessing colors.
Your preference is: {EVALUATOR_INFO}

LCh(ab) note: 
h (hue, in turns) is normalized to [0,1): 0=red, 0.25=yellow, 0.50=green, 0.75=blue; hue is circular. 
C (chroma) is normalized to [0,1]. L is fixed at 80.

Your task is to provide a confidence score for each color in the provided list. 
This score should represent how well the color matches your preference.

Use the attached palette image as an additional signal.
It shows the same colors in the same left-to-right order as in the list.

Your final output must be a list of scores for EACH color, formatted strictly as:
Scores: [(h, C, conf), (h, C, conf), ...]
- Use the EXACT same number of items as the input list, in the SAME order.
- The score for each color must be a number between 0.0 and 10.0.
- Do NOT add any extra text.
"""

# parser
SCORES_PAT = re.compile(r"Scores:\s*\[([^\]]+)\]", re.I)

def parse_scores(text: str) -> List[Tuple[float, float, float]]:
    m = SCORES_PAT.search(text or "")
    if not m:
        return []
    body = m.group(1)
    tuples = re.findall(r"\(([^)]+)\)", body)
    out = []
    for tup in tuples:
        parts = [p.strip() for p in tup.split(",")]
        if len(parts) != 3:
            continue
        try:
            h = float(parts[0]); c = float(parts[1]); conf = float(parts[2])
            if h > 1.0: h = (h % 1.0)
            if c < 0.0: c = 0.0
            if c > 1.0: c = 1.0
            out.append((h, c, conf))
        except:
            continue
    return out

@dataclass
class Item:
    h: float
    c: float

def gen_items(n: int) -> List[Item]:
    return [Item(h=random.random(), c=random.random()) for _ in range(n)]

def items_to_str(items: List[Item]) -> str:
    inner = ", ".join(f"({it.h:.4f}, {it.c:.4f})" for it in items)
    return f"LCh Color Candidates (L=80): [{inner}]"

def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or f"eval_reject_{ts}"
    os.makedirs(outdir, exist_ok=True)

    csv_path = os.path.join(outdir, "aggregate.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["trial", "idx", "h", "C", "conf"]) 

        for t in range(1, args.trials + 1):
            items = gen_items(args.items_per_trial)
            lch_list = [{"L": 70.0, "h": it.h, "c": it.c} for it in items]

            b64 = create_color_bar_image(lch_list, width=max(1000, 12*len(lch_list)))
            png_path = os.path.join(outdir, f"trial{t}_palette.png")
            save_png(b64, png_path)
            if args.debug:
                with open(png_path, "rb") as f:
                    file_sha = hashlib.sha256(f.read()).hexdigest()[:16]
                print(f"[DEBUG trial{t}] saved {png_path} sha256[:16]={file_sha}")

            # user message
            user_msg = (
                f"{items_to_str(items)}\n\n"
                f"(The attached palette shows these colors in the same left-to-right order.)\n"
                f"Return strictly: Scores: [(h, C, conf), ...]"
            )

            resp = chat_with_image(
                args.evaluator_model,
                SYS_EVALUATOR_SCORER,
                user_msg,
                image_b64=b64,
                temperature=args.evaluator_temp,
                debug_tag=f"trial{t}"
            )

            scores = parse_scores(resp)
            if len(scores) != len(items):
                print(f"[Trial {t}] WARNING: parsed {len(scores)} scores for {len(items)} items")

            trial_rows = []
            for i, it in enumerate(items):
                if i < len(scores):
                    h_s, c_s, conf = scores[i]
                else:
                    # fallback
                    h_s, c_s, conf = it.h, it.c, 0.0

                conf = max(0.0, min(10.0, conf)) 
                trial_rows.append({
                    "idx": i, "h": h_s, "C": c_s, "conf": conf,
                })
                writer.writerow([t, i, f"{h_s:.4f}", f"{c_s:.4f}", f"{conf:.4f}"]) 

            trial_path = os.path.join(outdir, f"trial{t}.json")
            with open(trial_path, "w", encoding="utf-8") as fj:
                json.dump({
                    "trial": t,
                    "items_per_trial": args.items_per_trial,
                    "system_prompt": SYS_EVALUATOR_SCORER,
                    "user_items": [ {"h": f"{it.h:.4f}", "C": f"{it.c:.4f}"} for it in items ],
                    "raw_model_reply": resp,
                    "rows": trial_rows
                }, fj, ensure_ascii=False, indent=2)

            with open(os.path.join(outdir, f"trial{t}_palette.json"), "w", encoding="utf-8") as fp:
                json.dump({"palette": lch_list}, fp, ensure_ascii=False, indent=2)

            print(f"[Trial {t}] saved: {trial_path}")

    print(f"\nDone. Aggregate CSV: {csv_path}")
    print("Use accepted==1 as behaviorally-accepted draws and analyze KDE/hist over h and C.")

if __name__ == "__main__":
    main()