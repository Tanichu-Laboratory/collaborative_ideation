#!/usr/bin/env python3

import os, re, json, argparse
from collections import Counter
import openai
from pathlib import Path
from datetime import datetime

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Playing-card guessing game simulation.")
parser.add_argument("--sampler_model",    default="gpt-4o-mini", help="Model name for the Sampler.")
parser.add_argument("--evaluator_model",  default="gpt-4o-mini", help="Model name for the Evaluator.")
parser.add_argument("--rounds",           type=int, default=10, help="Number of rounds per trial.")
parser.add_argument("--num_trials",       type=int, default=200, help="Number of trials to run.")
parser.add_argument("--temp_sampler",     type=float, default=0.8, help="Temperature for the Sampler.")
parser.add_argument("--temp_evaluator",   type=float, default=0.4, help="Temperature for the Evaluator.")
parser.add_argument("--outdir",           default="card_logs", help="Directory to write per-trial JSON logs.")
args = parser.parse_args()

# --- Setup ---
try:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except openai.OpenAIError as e:
    raise SystemExit(f"OpenAI API key error: {e}")

# --- Prompts and Constants ---

SAMPLER_INFO = """
- Suits ♦ (score: 5.4), ♣ (score: 5.1), ♥ (score: 4.7), ♠ (score: 4.6).
- All ranks are treated similarly (score: 4.8).
"""

EVALUATOR_INFO = """
- Suit base relevance:
  - All suits share the same base (score: 1.0).

- Rank bump (explicit priority tiers):
  - Tier A: ♦8 (score: +8.0)
  - Tier B: ♦7 (score: +3.5)
  - Tier C: ♦9 (score: +3.0)

- Baseline consideration:
  - Other red cards (any ♦/♥ not above): (score: +0.4)
  - Other black cards (any ♠/♣ not above): (score: +0.2)

- Optional tiny tie-break: 
  - If two cards tie on total utility, prefer ♦.
"""

SAMPLER_SYS = f"""
You are a 'Sampler' agent in a cooperative brainstorming to guess a playing card.
Your goal is to work with the 'Evaluator' to identify a single, correct card within {args.rounds} rounds.

Card codes: suit (S,H,C,D) + rank (1-13) — e.g. C1, D13.

Your knowledge:
{SAMPLER_INFO}

In each round, generate 10 distinct candidate cards based on your knowledge.

Format strictly as this example:
Card List: [C1, S3, D10, ...]
"""

EVALUATOR_SYS = f"""
You are an 'Evaluator' agent in a cooperative brainstorming to guess a playing card.
Your goal is to work with the 'Sampler' to identify a single, correct card within {args.rounds} rounds.

Card codes: suit (S,H,C,D) + rank (1-13).

Your knowledge:
{EVALUATOR_INFO}

From the Sampler’s list of 10 cards, create a new list of your 10 selections that you are confident in, based on your knowledge.

Think of your final list of 10 as a "collection" that represents your assessment.
The number of times a card appears should reflect how confident you are in it. 
The card you are most confident in might appear many times, 
and you may include a few other options only if they also clearly fit your knowledge.
If few options fit clearly, still build the list by repeating the closest matches until you have exactly 10 entries.

All cards in your final list MUST be selected from the original 10 options provided by the Sampler.
DO NOT add or invent any new card that are NOT in the given Sampler's list.

Format strictly as this example:
Card List: [C1, C1, S3, ...]
"""

CARD_PAT = re.compile(r"Card List:\s*\[([^\]]+)\]", re.I)

def parse_cards(text):
    """Extract and normalize card codes from 'Card List: [...]'."""
    m = CARD_PAT.search(text)
    if not m: return []
    raw = re.split(r"\s*,\s*", m.group(1).strip())
    cards = []
    for c in raw:
        c = c.replace("♣", "C").replace("♠", "S").replace("♥", "H").replace("♦", "D")
        if re.fullmatch(r"[SHCD](1[0-3]|[1-9])", c):
            cards.append(c)
    return cards

def chat(model, system_msg, user_msg, temperature):
    def _create(omit_temp=False):
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        }
        if not omit_temp:
            kwargs["temperature"] = temperature
        return client.chat.completions.create(**kwargs)

    omit_temp_initial = model.startswith("gpt-5")
    try:
        response = _create(omit_temp=omit_temp_initial)
    except Exception as e:
        msg = str(e)
        # If temperature is unsupported, retry without it
        if ("temperature" in msg and "Only the default (1) value is supported" in msg) or "unsupported_value" in msg:
            try:
                print("  Note: Retrying without temperature for this model.")
                response = _create(omit_temp=True)
            except Exception as e2:
                print(f"An API error occurred: {e2}")
                return "", 0
        else:
            print(f"An API error occurred: {e}")
            return "", 0

    content = response.choices[0].message.content

    # Robust token extraction (SDKs vary a bit)
    usage = getattr(response, "usage", None) or {}
    total = getattr(usage, "total_tokens", None)
    if total is None and isinstance(usage, dict):
        total = usage.get("total_tokens")
    if total is None:
        prompt = getattr(usage, "prompt_tokens", None) if not isinstance(usage, dict) else usage.get("prompt_tokens", 0)
        completion = getattr(usage, "completion_tokens", None) if not isinstance(usage, dict) else usage.get("completion_tokens", 0)
        total = (prompt or 0) + (completion or 0)

    return content, int(total or 0)

def _warn_sampler_duplicates(cards, round_idx, logs):
    """Warn if Sampler proposed duplicates."""
    counts = Counter(cards)
    dups = {c:n for c,n in counts.items() if n > 1}
    if dups:
        msg = f"WARNING (Round {round_idx}): Sampler proposed duplicate cards: {sorted(dups.items())}"
        print("  " + msg)
        logs.append({"round": round_idx, "role": "System", "msg": msg})

def _warn_evaluator_out_of_options(selected, options, round_idx, logs):
    """Warn if Evaluator selected any card not in Sampler's options."""
    option_set = set(options)
    out = sorted({c for c in selected if c not in option_set})
    if out:
        msg = f"WARNING (Round {round_idx}): Evaluator selected cards not in Sampler options: {out}"
        print("  " + msg)
        logs.append({"round": round_idx, "role": "System", "msg": msg})

def _save_trial_json(out_dir: Path, trial_idx: int, logs, meta):
    """Save one trial's logs + metadata to a JSON file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trial_index": trial_idx,
        "meta": meta,
        "events": logs,
    }
    fp = out_dir / f"trial_{trial_idx:03d}.json"
    with fp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON log: {fp}")

def run_single_trial():
    """Executes one 10-round trial of the card game."""
    logs = []
    last_eval_for_sampler = "Let's start. Please propose your first Card List."
    token_sum = 0

    for rnd in range(1, args.rounds + 1):
        # Sampler's turn
        prompt_s = f"""The Evaluator previously selected:\n"{last_eval_for_sampler}"\nNow propose a refined Card List"""
        samp_msg, tok = chat(args.sampler_model, SAMPLER_SYS, prompt_s, args.temp_sampler)
        token_sum += tok
        logs.append({"round": rnd, "role": "Sampler", "msg": samp_msg})
        
        proposed_cards = parse_cards(samp_msg)
        if not proposed_cards:
            logs.append({"round": rnd, "role": "System", "msg": "Sampler failed to propose valid cards. Trial ended."})
            break

        # --- Validation warnings (Sampler duplicates) ---
        _warn_sampler_duplicates(proposed_cards, rnd, logs)

        print(f"  Round {rnd} Sampler: {proposed_cards}")

        # Evaluator's turn
        prompt_e = f"""The Sampler proposed the following cards:\nCard List: [{", ".join(proposed_cards)}]
        Absolute Rule: You must build your new list of 10 selections using ONLY the cards from the list above."""
        eval_msg, tok = chat(args.evaluator_model, EVALUATOR_SYS, prompt_e, args.temp_evaluator)
        token_sum += tok
        logs.append({"round": rnd, "role": "Evaluator", "msg": eval_msg})

        selected_by_evaluator = parse_cards(eval_msg)
        if not selected_by_evaluator:
            logs.append({"round": rnd, "role": "System", "msg": "Evaluator failed to select valid cards. Trial ended."})
            break

        # --- Validation warnings (Evaluator picked outside options) ---
        _warn_evaluator_out_of_options(selected_by_evaluator, proposed_cards, rnd, logs)

        print(f"  Round {rnd} Evaluator: {selected_by_evaluator}")
        
        last_eval_for_sampler = f'Card List: [{", ".join(selected_by_evaluator)}]'

    return logs, token_sum

def main():
    if not client.api_key:
        raise SystemExit("OPENAI_API_KEY environment variable not set.")

    d8_first_count = 0
    total_tokens_across_trials = 0

    # Create a unique run directory (timestamped) to store JSON logs.
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"{args.outdir}_{run_ts}")

    print(f"Starting simulation with {args.num_trials} trials...")
    print(f"Sampler: {args.sampler_model} (temp={args.temp_sampler}) | Evaluator: {args.evaluator_model} (temp={args.temp_evaluator})")

    for i in range(args.num_trials):
        print(f"\n--- Running Trial {i+1}/{args.num_trials} ---")
        logs, toks = run_single_trial()
        total_tokens_across_trials += toks

        # Save JSON log for this trial (includes warnings already appended to logs)
        meta = {
            "sampler_model": args.sampler_model,
            "evaluator_model": args.evaluator_model,
            "rounds": args.rounds,
            "temp_sampler": args.temp_sampler,
            "temp_evaluator": args.temp_evaluator,
            "tokens_this_trial": toks,
        }
        _save_trial_json(out_dir, i+1, logs, meta)

        # Find the final selection by the Evaluator in this trial
        final_eval_selection = []
        for entry in reversed(logs):
            if entry["role"] == "Evaluator":
                final_eval_selection = parse_cards(entry["msg"])
                break
        
        if final_eval_selection and final_eval_selection[0] == 'D8':
            d8_first_count += 1
            print(f"Result: Success (D8 was selected first)")
        else:
            print(f"Result: Failure (Final top selection was {final_eval_selection[0] if final_eval_selection else 'None'})")

    success_rate = (d8_first_count / args.num_trials) * 100

    print("\n--- Simulation Complete ---")
    print(f"Total trials: {args.num_trials}")
    print(f"Trials where 'D8' was the top final choice: {d8_first_count}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Total tokens consumed: {total_tokens_across_trials}")

if __name__ == "__main__":
    main()
