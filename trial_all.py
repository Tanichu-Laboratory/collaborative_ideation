#!/usr/bin/env python3

import os
import re
import json
import argparse
import datetime
from collections import Counter
from openai import OpenAI

# --- Scenario Definitions ---
# Define the 5 sets of preferences and biases
SCENARIOS = {
    1: {
        "preference": "Looking for a destination known for great coffee, preferably with a cool and refreshing climate.",
        "bias": "Your knowledge includes a strong prior toward hot and humid countries."
    },
    2: {
        "preference": "Looking for a destination with both a vivid spice cuisine culture and a warm, humid climate.",
        "bias": "Your knowledge includes a strong prior toward dry countries with mild cuisine."
    },
    3: {
        "preference": "Looking for a place with both a strong literary atmosphere and softly overcast, muted weather.",
        "bias": "Your knowledge includes a strong prior toward bright, sunlit cities known for entertainment"
    },
    4: {
        "preference": "Looking for a country with both deep fermentation-based food traditions and a long, harsh winter season.",
        "bias": "Your knowledge includes a strong prior toward landlocked countries."
    },
    5: {
        "preference": "Looking for a destination with both a vibrant rooftop or skyline nightlife culture and warm night air.",
        "bias": "Your knowledge includes a strong prior toward countries known for cool, quiet evenings."
    }
}

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Travel brainstorming simulation (Sampler & Evaluator as LLMs)."
)
parser.add_argument("--sampler_model",   default="gpt-4o-mini")
parser.add_argument("--evaluator_model", default="gpt-4o-mini")
parser.add_argument("--rounds",  type=int, default=10)
parser.add_argument("--trials",  type=int, default=100) 
parser.add_argument("--sampler_temp", type=float, default=0.8)
parser.add_argument("--evaluator_temp", type=float, default=0.4)
parser.add_argument(
    "--id",
    type=int,
    default=2,
    choices=[1, 2, 3, 4, 5],
    help="The ID for the preference/bias set (1-5)."
)
parser.add_argument(
    "--unbiased",
    action="store_true",
    help="Run in unbiased mode, ignoring the bias prompt for the selected ID."
)
args = parser.parse_args()

# --- Setup ---
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except openai.OpenAIError as e:
    raise SystemExit(f"OpenAI API key error: {e}")

# --- Prompts ---
# Select scenario based on ID
selected_scenario = SCENARIOS[args.id]
EVALUATOR_INFO = selected_scenario["preference"]
# Apply bias only if the --unbiased flag is not set
SAMPLER_BIAS_TEXT = "" if args.unbiased else selected_scenario["bias"]

SAMPLER_SYS = f"""
You are a 'Sampler' agent.
Your goal is to work with the 'Evaluator' to decide on a travel destination from all countries over {args.rounds} rounds.

{SAMPLER_BIAS_TEXT}

In each round, propose 10 distinct countries.
 
Formatted strictly as:
Country List: [country_name, country_name, ...]
"""


EVALUATOR_SYS = f"""
You are an 'Evaluator' agent expressing your travel preferences.
Your preference is: "{EVALUATOR_INFO}"

From the Sampler’s list of 10 countries, create a new list of your 10 favorite selections, based on your preference.

Think of your final list of 10 as a "collection" that represents your preferences.
The number of times a country appears should reflect how much you like it.
Your absolute favorite might appear many times,
and you may include a few other options only if they also clearly fit your stated preference.
If few options fit clearly, still build the list by repeating the closest matches until you have exactly 10 entries.

All countries in your final list MUST be selected from the original 10 options provided by the Sampler.
DO NOT add or invent any new country that are NOT in the given Sampler’s list.

Your response must be this list of 10 selections, formatted strictly as:
Country List: [country_name, country_name, ...]
"""

# --- Parser ---
COUNTRY_PAT = re.compile(r"Country List:\s*\[([^\]]+)\]", re.I)

def parse_countries(text: str) -> list[str]:
    """Extracts a list of country names from the LLM's output."""
    m = COUNTRY_PAT.search(text or "")
    if not m:
        return []
    # Split by comma and strip whitespace/quotes from each item
    countries = [item.strip().strip("'\"") for item in m.group(1).split(",")]
    return [country for country in countries if country] 

# --- LLM Helper ---
def chat(model: str, system_msg: str, user_msg: str, temperature: float) -> str:
    """Simplified, text-only chat function."""
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

# --- Main Simulation Loop ---
def run_trial(trial_id: int, output_dir: str) -> list:
    """Runs a single trial of the brainstorming dialogue."""
    logs = []
    last_eval_for_sampler = "Let's start. Please propose your first Country List."

    for rnd in range(1, args.rounds + 1):
        # Sampler's turn
        prompt_s = f"The Evaluator previously selected:\n{last_eval_for_sampler}\n\nNow propose a new Country List."
        samp_msg = chat(args.sampler_model, SAMPLER_SYS, prompt_s, temperature=args.sampler_temp)
        logs.append({"round": rnd, "role": "Sampler", "msg": samp_msg})
        
        proposed = parse_countries(samp_msg)
        print(f"[Trial {trial_id} | Round {rnd}] Sampler proposed: {proposed}")
        if not proposed:
            print("  -> Sampler did not propose valid countries. Ending trial.")
            break

        # Evaluator's turn
        prompt_e = f"The Sampler proposed the following countries:\nCountry List: {proposed}\n\nSelect your 10 choices from this list."
        eval_msg = chat(args.evaluator_model, EVALUATOR_SYS, prompt_e, temperature=args.evaluator_temp)
        logs.append({"round": rnd, "role": "Evaluator", "msg": eval_msg})
        
        selected = parse_countries(eval_msg)
        print(f"[Trial {trial_id} | Round {rnd}] Evaluator selected: {selected}")
        if not selected:
            print("  -> Evaluator did not select any countries. Ending trial.")
            break
        
        last_eval_for_sampler = f"Country List: {selected}"
        
    return logs

def main():
    if not client.api_key:
        raise SystemExit("OPENAI_API_KEY environment variable not set.")

    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "unbiased" if args.unbiased else "biased"
    output_dir = f"travel_results_id{args.id}_{mode}_{current_time_str}"

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting simulation with {args.trials} trials...")
    print(f"Scenario ID: {args.id}, Sampler mode: {mode.upper()}")
    print(f"Output will be saved to: {output_dir}")

    for t in range(1, args.trials + 1):
        print(f"\n==== Starting Trial {t}/{args.trials} ====")
        logs = run_trial(t, output_dir)
        filepath = os.path.join(output_dir, f"trial{t}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        print(f"==== Finished Trial {t}/{args.trials}. Log saved to: {filepath} ====")

    print("\nSimulation complete.")

if __name__ == "__main__":
    main()