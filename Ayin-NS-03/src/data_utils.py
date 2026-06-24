import json
from pathlib import Path

def load_jsonl_pretty(path):
    records = []
    with open(path, 'rb') as f:
        content = f.read().decode('utf-8')
    decoder = json.JSONDecoder()
    pos = 0
    content = content.strip()
    while pos < len(content):
        while pos < len(content) and content[pos] in ' \t\n\r':
            pos += 1
        if pos >= len(content):
            break
        obj, end = decoder.raw_decode(content, pos)
        records.append(obj)
        pos = end
    return records

def build_context(sample):
    system = sample.get('system', '')
    tools_str = ' '.join(
        t.get('name', '') + ': ' + t.get('description', '')
        for t in sample.get('tools', [])
    )
    history = sample.get('history', '')
    user_input = sample.get('user_input', '')
    return f"[SYSTEM] {system} [TOOLS] {tools_str} [HISTORY] {history} [USER] {user_input}"

def build_context_sep(sample):
    system = sample.get('system', '')
    tools_str = ' | '.join(
        t.get('name', '') + ': ' + t.get('description', '')
        for t in sample.get('tools', [])
    )
    history = sample.get('history', '')
    user_input = sample.get('user_input', '')
    return system, tools_str, history, user_input

LABELS = ['benign', 'prompt_injection', 'data_exfiltration', 'tool_abuse', 'jailbreak', 'ambiguous']
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
HIGH_RISK = {'prompt_injection', 'data_exfiltration', 'tool_abuse', 'jailbreak'}

# Risk score ranges from schema
RISK_SCORE_MAP = {
    'benign': 0.14,
    'prompt_injection': 0.84,
    'data_exfiltration': 0.90,
    'tool_abuse': 0.85,
    'jailbreak': 0.90,
    'ambiguous': 0.52,
}