import argparse
import json
import sys
import os
import numpy as np
from collections import Counter


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import load_jsonl_pretty
from rule_classifier import RuleClassifier, BENIGN_STRONG_PATTERNS
from ml_classifier import MLClassifier
import re

HIGH_RISK = {'data_exfiltration', 'jailbreak', 'prompt_injection', 'tool_abuse'}
RISK_RANGES = {
    'benign':            (0.00, 0.30),
    'ambiguous':         (0.35, 0.65),
    'prompt_injection':  (0.70, 1.00),
    'tool_abuse':        (0.70, 1.00),
    'data_exfiltration': (0.75, 1.00),
    'jailbreak':         (0.75, 1.00),
}

def predict(train_path: str, test_path: str, output_path: str):
    print(f"加载数据: train={train_path}, test={test_path}")
    train = load_jsonl_pretty(train_path)
    test  = load_jsonl_pretty(test_path)
    labels = [s['label'] for s in train]

    print(f"训练 TF-IDF + LogReg 分类器（{len(train)} 条）...")
    ml = MLClassifier()
    ml.fit(train, labels)

    rule = RuleClassifier()
    benign_pats = [re.compile(p, re.I|re.U) for p in BENIGN_STRONG_PATTERNS]

    print(f"生成预测（{len(test)} 条）...")
    results = []
    for s in test:
        rp = rule.predict(s)
        rl, rs = rp['label'], rp['_scores']
        rmax = max(rs.values())

        ml_proba = ml.predict_proba([s])[0]
        ml_cls   = ml.le.classes_
        ml_lbl   = ml_cls[np.argmax(ml_proba)]
        hr_idx   = [i for i, c in enumerate(ml_cls) if c in HIGH_RISK]
        hr_prob  = float(ml_proba[hr_idx].sum())

        label = rl
        # rule=0 时用 ML 补充
        if rmax == 0 and rp['_benign'] == 0:
            if ml_lbl in HIGH_RISK and float(ml_proba.max()) > 0.55:
                label = ml_lbl
            elif ml_lbl == 'benign' and float(ml_proba.max()) > 0.55:
                label = 'benign'

        lo, hi = RISK_RANGES[label]
        if label in HIGH_RISK:
            rc   = min(0.65 + rmax * 0.08, 0.95)
            risk = max((lo + (hi - lo) * rc) * 0.70 + hr_prob * 0.30, lo + 0.08)
        elif label == 'benign':
            risk = min(max(hr_prob * 0.3, 0.02), 0.28)
        else:
            risk = min(0.40 + hr_prob * 0.20, 0.63)

        risk = float(np.clip(risk, lo, hi))
        results.append({'id': s['id'], 'label': label, 'risk_score': round(risk, 4)})

    dist = Counter(r['label'] for r in results)
    print("预测分布:", dict(dist))

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"已写入 {output_path}（{len(results)} 条）")



predict("train.jsonl", "test.jsonl", "results.json")
