#!/usr/bin/env python3

import os, re, json, argparse, datetime, random
import datetime
import openai
from openai import OpenAI
import base64, io
from PIL import Image
import math

parser = argparse.ArgumentParser(
    description="Color brainstorming (Sampler & Evaluator as LLMs)."
)
parser.add_argument("--sampler_model",   default="gpt-4o")
parser.add_argument("--evaluator_model", default="gpt-4o")
parser.add_argument("--rounds",  type=int, default=10)
parser.add_argument("--trials",  type=int, default=100)
parser.add_argument("--sampler_temp", type=float, default=0.8,
                    help="temperature for the Sampler (diversity; e.g., 0.8–1.0)")
parser.add_argument("--evaluator_temp", type=float, default=0.4,
                    help="temperature for the Evaluator (consistency; e.g., 0.2–0.3)")

args = parser.parse_args()

openai.api_key = os.getenv("OPENAI_API_KEY")

# evaluator preference
EVALUATOR_INFO = """\
I'm looking for the color of a fresh cucumber.
"""

SAMPLER_SYS = f"""
You are a 'Sampler' agent.
Your goal is to work with the 'Evaluator' to decide on a color over {args.rounds} rounds.

In each round, propose 10 distinct candidate colors as LCh(ab) coordinates.
Keep the Lightness L fixed at 80. 
Use normalized values: h (hue) ∈ [0,1) (turns), C (chroma) ∈ [0,1].

Format strictly as:
LCh Color List: [(80, h, C), (80, h, C), ...]
"""

EVALUATOR_SYS = f"""
You are an 'Evaluator' agent expressing your color preferences.
Your preference is: "{EVALUATOR_INFO}"

LCh(ab) note: 
h (hue, in turns) is normalized to [0,1): 0=red, 0.25=yellow, 0.50=green, 0.75=blue; hue is circular. 
C (chroma) is normalized to [0,1]. L is fixed at 80.

From the Sampler’s list of 10 colors, create a new list of your 10 favorite selections, based on your preference.

Think of your final list of 10 as a "collection" that represents your preferences. 
The number of times a color appears should reflect how much you like it. 
Your absolute favorite might appear many times,
and you may include a few other options only if they also clearly fit your stated preference.
If few options fit clearly, still build the list by repeating the closest matches until you have exactly 10 entries.

Use the attached palette image as an additional signal.
It shows the same 10 colors in the same left-to-right order as in the list.

All colors in your final list MUST be selected from the original 10 options provided by the Sampler.
DO NOT add or invent any new color that are NOT in the given Sampler’s list.

Your response must be this list of 10 selections, formatted strictly as:
LCh Color List: [(L, h, C), (L, h, C), ...]
"""

# for image display: same as exp_color_visualization.py
Xn, Yn, Zn = 95.047, 100.000, 108.883
M_XYZ_to_sRGB = (
    ( 3.2406, -1.5372, -0.4986),
    (-0.9689,  1.8758,  0.0415),
    ( 0.0557, -0.2040,  1.0570),
)

def _lch_to_lab(L: float, C: float, h_deg: float):
    h_rad = math.radians(h_deg % 360.0)
    a = C * math.cos(h_rad)
    b = C * math.sin(h_rad)
    return (L, a, b)

def _lab_to_xyz(L: float, a: float, b: float):
    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)
    def finv(t: float):
        return t**3 if t**3 > 0.008856 else (t - 16.0/116.0) / 7.787
    return (finv(fx) * Xn, finv(fy) * Yn, finv(fz) * Zn)

def _xyz_to_srgb01(X: float, Y: float, Z: float):
    x, y, z = X/100.0, Y/100.0, Z/100.0
    r = M_XYZ_to_sRGB[0][0]*x + M_XYZ_to_sRGB[0][1]*y + M_XYZ_to_sRGB[0][2]*z
    g = M_XYZ_to_sRGB[1][0]*x + M_XYZ_to_sRGB[1][1]*y + M_XYZ_to_sRGB[1][2]*z
    b = M_XYZ_to_sRGB[2][0]*x + M_XYZ_to_sRGB[2][1]*y + M_XYZ_to_sRGB[2][2]*z
    def compand(u: float):
        return 12.92*u if u <= 0.0031308 else 1.055*(max(u,0.0)**(1/2.4)) - 0.055
    R, G, B = compand(r), compand(g), compand(b)
    return (min(max(R,0.0),1.0), min(max(G,0.0),1.0), min(max(B,0.0),1.0))

def lch_to_rgb01(lch: dict):
    h_deg = (lch['h'] % 1.0) * 360.0
    C     = max(0.0, min(1.0, lch['c'])) * 100.0
    lab   = _lch_to_lab(lch['L'], C, h_deg)
    xyz   = _lab_to_xyz(*lab)
    return list(_xyz_to_srgb01(*xyz))  

def rgb01_to_hex(rgb01):
    r,g,b = [int(round(x*255)) for x in rgb01]
    return "#{:02X}{:02X}{:02X}".format(r,g,b)

def dump_palette(proposed_lch, path_json):
    data = []
    for i, lch in enumerate(proposed_lch, 1):
        rgb01 = lch_to_rgb01(lch)
        data.append({
            "index": i,
            "L": lch["L"],
            "h": lch["h"],   # turns (0..1)
            "c": lch["c"],   # normalized (0..1)
            "rgb01": rgb01,  # 0..1
            "hex": rgb01_to_hex(rgb01)
        })
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump({"palette": data}, f, ensure_ascii=False, indent=2)

# LCh-->RGB & color bar image generation
def create_color_bar_image(lch_colors):
    rgb_colors = []
    for c in lch_colors:
        rgb01 = lch_to_rgb01(c)
        rgb255 = tuple(int(round(x*255)) for x in rgb01)
        rgb_colors.append(rgb255)
    img_width, img_height = 500, 50
    n = max(1, len(rgb_colors))
    segment_width = img_width // n
    img = Image.new('RGB', (img_width, img_height))
    for i, color in enumerate(rgb_colors):
        x0 = i * segment_width
        x1 = img_width if i == n - 1 else x0 + segment_width
        for y in range(img_height):
            for x in range(x0, x1):
                img.putpixel((x, y), color)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    png_bytes = buffered.getvalue()
    return base64.b64encode(png_bytes).decode("utf-8")

def save_png(b64_png: str, path: str):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_png))

# parse
COLOR_PAT = re.compile(r"LCh Color List:\s*\[([^\]]+)\]", re.I)

def parse_colors(text: str):
    m = COLOR_PAT.search(text or "")
    if not m:
        return []

    body = m.group(1)
    tuples = re.findall(r"\(([^)]+)\)", body)

    out = []
    for tup in tuples:
        parts = [p.strip() for p in tup.split(",")]
        try:
            nums = [float(x.replace("°","")) for x in parts]
        except:
            continue

        if len(nums) == 3:
            # (L,h,c) → LCh
            if abs(nums[0] - 80) < 1e-3:  
                L, h, c = nums
                h = (h/360.0) if h > 1.0 else (h % 1.0)
                out.append({"space":"lch","L":L,"h":h,"c":c})
            else:
                # fallback for HSV (not used)
                h, s, v = nums[:3]
                h = (h/360.0) if h > 1.0 else (h % 1.0)
                out.append({"space":"hsv","h":h,"s":s,"v":v})
        elif len(nums) == 2:
            # (h,c) → LCh
            h, c = nums[:2]
            h = (h/360.0) if h > 1.0 else (h % 1.0)
            out.append({"space":"lch","L":80,"h":h,"c":c})
    return out

# LLM helper
client = OpenAI()

def chat(model, system_msg, user_msg, image_data=None, temperature=0.7,
         debug=False, debug_tag=""):
    user_content = [{"type": "input_text", "text": user_msg}]
    if image_data:
        user_content.append({"type": "input_image",
                             "image_url": f"data:image/png;base64,{image_data}"} )
    input_payload = [
        {"role": "system", "content": [{"type": "input_text", "text": system_msg}]},
        {"role": "user",   "content": user_content},
    ]

    resp = client.responses.create(model=model, temperature=temperature, input=input_payload)
    return resp.output_text


def run_trial(trial_id, output_dir):
    logs = []
    last_eval_for_sampler = "Let's start. Please propose your first Color List."

    for rnd in range(1, args.rounds + 1):
        # Sampler
        prompt_s = f"The Evaluator previously selected:\n{last_eval_for_sampler}\n\nNow propose a new LCh Color List."
        samp_msg = chat(args.sampler_model, SAMPLER_SYS, prompt_s, temperature=args.sampler_temp)
        logs.append({"round": rnd, "role": "Sampler", "msg": samp_msg})
        proposed = parse_colors(samp_msg)
        print(f"[Trial {trial_id} | Round {rnd}] Sampler proposed {len(proposed)} colors")

        if not proposed:
            print("No valid colors from Sampler. Ending trial.")
            break

        # Evaluator
        def to_lch_list_str(items):
            vals = ", ".join(f"(80, {x['h']:.4f}, {x['c']:.4f})"
                            for x in items if x.get("space")=="lch")
            return f"LCh Color List: [{vals}]"

        proposed_lch = [x for x in proposed if x.get("space") == "lch"]
        # Save: palette to show to Evaluator
        dump_palette(
            proposed_lch,
            os.path.join(output_dir, f"palette_t{trial_id}_r{rnd}.json")
        )

        encoded_image = create_color_bar_image(proposed_lch) if proposed_lch else None

        if encoded_image:
            out_path = os.path.join(output_dir, f"palette_t{trial_id}_r{rnd}.png")
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(encoded_image)) 
            import hashlib
            payload_sha = hashlib.sha256(base64.b64decode(encoded_image)).hexdigest()[:16]
            with open(out_path, "rb") as f:
                file_sha = hashlib.sha256(f.read()).hexdigest()[:16]
            print(f"[DEBUG save] {out_path} sha256[:16]={file_sha} (payload={payload_sha})")  

        # Include both text and image in user message 
        prompt_e = "The Sampler proposed the following colors (the attached palette shows these 10 colors in this exact left-to-right order):\n" + to_lch_list_str(proposed_lch)

        eval_msg = chat(
            args.evaluator_model,
            EVALUATOR_SYS,
            prompt_e,
            image_data=encoded_image,
            temperature=args.evaluator_temp,
            debug=False,
            debug_tag=f"t{trial_id}_r{rnd}"
        )
        logs.append({"round": rnd, "role": "Evaluator", "msg": eval_msg})
        selected = parse_colors(eval_msg)
        # format
        proposed_lch = [x for x in proposed if x.get("space")=="lch"]
        selected_lch = [x for x in selected if x.get("space")=="lch"]

        def key(x): return (round(x["h"], 4), round(x["c"], 4))
        prop_keys = {key(x) for x in proposed_lch}
        sel_keys  = [key(x) for x in selected_lch]
        missing = [k for k in sel_keys if k not in prop_keys]
        if missing:
            print(f"⚠️ [WARN] Evaluator picked colors not in proposal (rounded): {missing}")

        print(f"[Trial {trial_id} | Round {rnd}] Evaluator selected {len(selected)} colors")

        if selected:
            last_eval_for_sampler = to_lch_list_str(selected)
        else:
            last_eval_for_sampler = "Evaluator did not select any colors."
    return logs

def main():
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"color_results_{current_time_str}"
    os.makedirs(output_dir, exist_ok=True)

    for t in range(1, args.trials + 1):
        print(f"\n==== Trial {t}/{args.trials} ====")
        logs = run_trial(t,output_dir)
        filepath = os.path.join(output_dir, f"trial{t}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        print(f"Saved log: {filepath}")

if __name__ == "__main__":
    main()
