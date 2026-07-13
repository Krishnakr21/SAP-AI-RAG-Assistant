# utils/agent.py
"""
A tiny, deterministic "agent" layer that can perform small actions
on retrieved documents. This is NOT a general LLM agent — it's a simple,
rule-based orchestrator you can explain easily in interviews.

Actions implemented:
- summarize: short summary of retrieved docs
- extract_financial_lines: return lines mentioning finance keywords
- run_report_simulation: produce a tiny "simulated report" (counts/keywords)
"""

from typing import List
import re

FINANCE_KEYWORDS = ["fi", "financial", "gl account", "posting", "invoice", "billing", "fiscal", "amount"]

def summarize_docs(docs: List[str], max_sentences: int = 3):
    # Very simple extractive summary: concatenate first N sentences across docs
    sentences = []
    for d in docs:
        for s in re.split(r'(?<=[.!?])\s+', d.strip()):
            if s:
                sentences.append(s.strip())
            if len(sentences) >= max_sentences:
                break
        if len(sentences) >= max_sentences:
            break
    return " ".join(sentences) if sentences else "No content to summarize."

def extract_financial_lines(docs: List[str]):
    results = []
    for d in docs:
        lines = re.split(r'[\n\r]+', d)
        for line in lines:
            low = line.lower()
            if any(keyword in low for keyword in FINANCE_KEYWORDS):
                results.append(line.strip())
    return results if results else ["No financial lines found in retrieved docs."]

def run_report_simulation(docs: List[str]):
    # Example: count occurrences of main SAP modules in retrieved docs
    counters = {"MM":0, "FI":0, "SD":0, "PP":0, "HR":0}
    for d in docs:
        low = d.lower()
        if "mm" in low or "material" in low: counters["MM"] += 1
        if "fi" in low or "financial" in low or "gl" in low: counters["FI"] += 1
        if "sd" in low or "sales" in low: counters["SD"] += 1
        if "pp" in low or "mrp" in low or "production" in low: counters["PP"] += 1
        if "hr" in low or "payroll" in low or "employee" in low: counters["HR"] += 1

    report_lines = [f"{k}: {v}" for k,v in counters.items()]
    return " | ".join(report_lines)
