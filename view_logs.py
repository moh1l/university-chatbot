"""
Quick viewer for logged student questions.

Usage:
    python view_logs.py
"""

import json
import os
from collections import Counter

LOG_PATH = "logs/questions.jsonl"


def main():
    if not os.path.exists(LOG_PATH):
        print("No questions logged yet. Ask something in the chat first.")
        return

    entries = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"Total questions logged: {len(entries)}\n")

    answered = sum(1 for e in entries if e["answered"])
    unanswered = len(entries) - answered
    print(f"Answered from data: {answered}")
    print(f"Not found in data (guardrail triggered): {unanswered}\n")

    print("--- All questions ---")
    for e in entries:
        status = "OK" if e["answered"] else "NOT FOUND"
        print(f"[{status}] {e['timestamp']}  {e['question']}")


if __name__ == "__main__":
    main()
