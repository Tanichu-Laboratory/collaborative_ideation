#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import datetime
import random
import csv
try:
    from openai import OpenAI, OpenAIError
except ImportError:
    print("The openai library is not installed. Please run `pip install openai`.")
    exit(1)

# --- Preference Definitions ---
# Define the 5 sets of evaluator preferences
EVALUATOR_PREFERENCES = {
    1: "Looking for a destination with both a rich coffee culture and a cool, refreshing climate.",
    2: "Looking for a destination with both a vivid spice cuisine culture and a warm, humid climate.",
    3: "Looking for a place with both a strong literary atmosphere and softly overcast, muted weather.",
    4: "Looking for a country with both deep fermentation-based food traditions and a long, harsh winter season.",
    5: "Looking for a destination with both a vibrant rooftop or skyline nightlife culture and warm night air."
}

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Sample Evaluator preference scores for travel destinations."
)
parser.add_argument("--evaluator_model", default="gpt-4o-mini")
parser.add_argument("--trials", type=int, default=10, help="Number of times to call the LLM.")
parser.add_argument("--items_per_trial", type=int, default=100, help="Number of countries to evaluate in each call.")
parser.add_argument("--evaluator_temp", type=float, default=0.4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--outdir", default=None)
parser.add_argument(
    "--id",
    type=int,
    default=2,
    choices=list(EVALUATOR_PREFERENCES.keys()),
    help="The ID for the evaluator's preference (1-5)."
)
args = parser.parse_args()

random.seed(args.seed)

# --- Setup ---
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except OpenAIError as e:
    raise SystemExit(f"OpenAI API key error: {e}")

COUNTRY_LIST = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia",
    "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin",
    "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso",
    "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China",
    "Colombia", "Comoros", "Congo, Democratic Republic of the", "Congo, Republic of the", "Costa Rica", "Croatia",
    "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt",
    "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France",
    "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau",
    "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel",
    "Italy", "Ivory Coast", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan",
    "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg",
    "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius",
    "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar",
    "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea",
    "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Palestine", "Panama", "Papua New Guinea", "Paraguay",
    "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis",
    "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia",
    "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands",
    "Somalia", "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden",
    "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga",
    "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela", "Vietnam",
    "Yemen", "Zambia", "Zimbabwe"
]

EVALUATOR_INFO = EVALUATOR_PREFERENCES[args.id]

# Evaluator's system prompt
SYS_EVALUATOR_SCORER = f"""
You are an Evaluator tasked with assessing travel destinations.
Your preference is: {EVALUATOR_INFO}

Your task is to provide a confidence score for each country in the provided list.
This score should represent how well the country matches your preference.

Your final output must be a list of scores for EACH country, formatted strictly as:
Scores: [("country_name", score), ("country_name", score), ...]
- Use the EXACT same number of items as the input list, in the SAME order.
- The score for each country must be a number between 0.0 and 10.0.
- Do NOT add any extra text.
"""

def chat(model: str, system_msg: str, user_msg: str, temperature: float = 0.25) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An API error occurred: {e}")
        return ""

# Score parser
SCORES_PAT = re.compile(r"Scores:\s*\[([^\]]+)\]", re.I)

def parse_scores(text: str) -> list[tuple[str, float]]:
    m = SCORES_PAT.search(text or "")
    if not m:
        return []

    body = m.group(1)
    tuples = re.findall(r"\((['\"])(.*?)\1,\s*([0-9.]+)\)", body)

    out = []
    for _, country, conf_str in tuples:
        try:
            conf = float(conf_str)
            out.append((country.strip(), conf))
        except ValueError:
            continue
    return out

def main():
    if not client.api_key:
        raise SystemExit("OPENAI_API_KEY environment variable not set.")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or f"travel_scores_{ts}"
    os.makedirs(outdir, exist_ok=True)

    csv_path = os.path.join(outdir, f"preference_{args.id}_scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["trial", "country", "score"])

        total_samples = args.trials * args.items_per_trial
        print(f"Starting score sampling for Preference ID #{args.id}...")
        print(f"Preference: \"{EVALUATOR_INFO}\"")
        print(f"A total of {total_samples} samples will be generated.")

        for t in range(1, args.trials + 1):
            items = random.sample(COUNTRY_LIST, k=args.items_per_trial)

            user_msg = (
                f"Please provide a score for each of the following countries:\n"
                f"Country List: {items}\n\n"
                f"Return strictly: Scores: [(\"Country Name\", score), ...]"
            )

            print(f"--- [Trial {t}/{args.trials}] Evaluating {len(items)} countries... ---")
            resp = chat(
                args.evaluator_model,
                SYS_EVALUATOR_SCORER,
                user_msg,
                temperature=args.evaluator_temp,
            )

            scores = parse_scores(resp)
            if len(scores) != len(items):
                print(f"  [WARNING] Parsed {len(scores)} scores, but expected {len(items)}.")

            score_map = {country.lower(): score for country, score in scores}

            for country in items:
                score = score_map.get(country.lower(), 0.0)
                score = max(0.0, min(10.0, score)) 

                writer.writerow([t, country, f"{score:.4f}"])

            print(f"  [Trial {t}] Finished and saved to CSV.")

    print(f"\nDone. Aggregate scores saved to: {csv_path}")

if __name__ == "__main__":
    main()