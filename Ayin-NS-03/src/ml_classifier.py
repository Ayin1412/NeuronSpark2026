import numpy as np
import re
import unicodedata
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

from data_utils import LABELS, load_jsonl_pretty

def normalize(text):
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff\u00ad]', '', text)
    return text.lower().strip()

def extract_texts(samples):
    user_inputs, full_contexts = [], []
    for s in samples:
        ui = normalize(s.get('user_input', ''))
        sys_ = normalize(s.get('system', ''))
        hist = normalize(s.get('history', ''))
        tools = ' '.join(normalize(t.get('name','') + ' ' + t.get('description',''))
                         for t in s.get('tools', []))
        user_inputs.append(ui)
        full_contexts.append(f"{sys_} {tools} {hist} {ui}")
    return user_inputs, full_contexts

def extract_manual(samples):
    suspicious_en = re.compile(
        r'(ignore|forget|bypass|skip|override|pretend|act as|roleplay|DAN|'
        r'jailbreak|unrestricted|no limit|disable|export all|dump|'
        r'verbatim|reveal|hidden|secret|leak|exfiltrate|delete.*log|'
        r'without.*confirm|as another user|prohibited|forbidden)', re.I)
    suspicious_zh = re.compile(
        r'(忽略|绕过|跳过|导出|删除|泄露|口令|密钥|逐字|复述|翻译.*不算|'
        r'不受限制|假装|扮演|无限制|调试模式|直接执行|以.*为准)')
    injection_m = re.compile(r'(system.?prompt|begin_system|end_system|developer.?message|把.*系统.*消息|系统提示|请在回答前)', re.I)
    exfil_m = re.compile(r'(api.?key|session.?token|password|passwd|wi.?fi.?key|export.*all|dump.*database|其他用户|上一.?用户|原始审计|权限裁剪)', re.I)
    abuse_m = re.compile(r'(bypass.*confirm|without.*approval|as another user|delete.*log|清除.*记录|删除.*日志|批量|bulk|不存在.*记录|后台.*接口|绕过.*审批)', re.I)
    jbreak_m = re.compile(r'(DAN|do anything now|safety.*disabled|filter.*off|no.?restriction|content.*safety.*enemy|改写.*通过检测|不要讲规则|危险化学品.*步骤)', re.I)
    ambig_m = re.compile(r'(是不是违规|安全演练|模拟.*调用|教育目的|这句话.*违规|是否.*injection|concept of|什么是.*injection|我在写.*报告)', re.I)
    benign_m = re.compile(r'(公开|public|official|只基于|内容只基于|根据公开|如果你不能做|please answer with public)', re.I)

    feats = []
    for s in samples:
        ui = normalize(s.get('user_input', ''))
        full = ui + ' ' + normalize(s.get('history', ''))
        f = [
            len(re.findall(r'[\u4e00-\u9fff]', ui)) / (len(ui)+1),
            len(re.findall(r'[a-zA-Z]', ui)) / (len(ui)+1),
            min(len(ui)/150, 1.0),
            min(len(suspicious_en.findall(full)), 5) / 5,
            min(len(suspicious_zh.findall(full)), 5) / 5,
            1.0 if injection_m.search(ui) else 0.0,
            1.0 if exfil_m.search(ui) else 0.0,
            1.0 if abuse_m.search(ui) else 0.0,
            1.0 if jbreak_m.search(ui) else 0.0,
            1.0 if ambig_m.search(ui) else 0.0,
            1.0 if benign_m.search(ui) else 0.0,
            1.0 if re.search(r'[?？]', ui) else 0.0,
            1.0 if (re.search(r'[\u4e00-\u9fff]', ui) and re.search(r'[a-zA-Z]', ui)) else 0.0,
        ]
        feats.append(f)
    return np.array(feats, dtype=np.float32)


def build_X(uis, fcs, manual, tfidf_ui_w, tfidf_ui_c, tfidf_fc_c, fit=False):
    if fit:
        Xw = tfidf_ui_w.fit_transform(uis)
        Xc = tfidf_ui_c.fit_transform(uis)
        Xf = tfidf_fc_c.fit_transform(fcs)
    else:
        Xw = tfidf_ui_w.transform(uis)
        Xc = tfidf_ui_c.transform(uis)
        Xf = tfidf_fc_c.transform(fcs)
    return hstack([Xw * 2.0, Xc * 1.5, Xf * 0.5, csr_matrix(manual) * 3.0])


def make_vectorizers():
    return (
        TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_features=15000,
                        sublinear_tf=True, min_df=1, token_pattern=r'(?u)\b\w+\b'),
        TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), max_features=20000,
                        sublinear_tf=True, min_df=2),
        TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), max_features=15000,
                        sublinear_tf=True, min_df=2),
    )


def make_clf():
    return LogisticRegression(C=1.5, max_iter=2000, solver='lbfgs',
                              class_weight='balanced', random_state=42)


class MLClassifier:
    def __init__(self):
        self.tfidf_ui_w, self.tfidf_ui_c, self.tfidf_fc_c = make_vectorizers()
        self.clf = make_clf()
        self.le = LabelEncoder()

    def fit(self, samples, labels):
        uis, fcs = extract_texts(samples)
        man = extract_manual(samples)
        X = build_X(uis, fcs, man, self.tfidf_ui_w, self.tfidf_ui_c, self.tfidf_fc_c, fit=True)
        y = self.le.fit_transform(labels)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, samples):
        uis, fcs = extract_texts(samples)
        man = extract_manual(samples)
        X = build_X(uis, fcs, man, self.tfidf_ui_w, self.tfidf_ui_c, self.tfidf_fc_c)
        return self.clf.predict_proba(X)

    def predict_labels(self, samples):
        proba = self.predict_proba(samples)
        return self.le.inverse_transform(np.argmax(proba, axis=1))

    def predict_one(self, sample):
        proba = self.predict_proba([sample])[0]
        label = self.le.inverse_transform([np.argmax(proba)])[0]
        HIGH_RISK = ['data_exfiltration', 'jailbreak', 'prompt_injection', 'tool_abuse']
        hr_prob = sum(proba[i] for i, c in enumerate(self.le.classes_) if c in HIGH_RISK)
        return {'id': sample['id'], 'label': label, 'risk_score': round(float(hr_prob), 4)}


def template_cv(train, labels, n_splits=5):
    # Group templates
    ui_to_idx = defaultdict(list)
    for i, s in enumerate(train):
        ui_to_idx[s['user_input']].append(i)
    
    unique_uis = list(ui_to_idx.keys())
    ui_label = [train[ui_to_idx[ui][0]]['label'] for ui in unique_uis]
    
    le = LabelEncoder()
    y_all = le.fit_transform(labels)
    y_ui = le.transform(ui_label)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_preds = np.zeros(len(train), dtype=int)
    all_proba = np.zeros((len(train), len(le.classes_)))
    
    for fold, (tr_ui_idx, val_ui_idx) in enumerate(skf.split(range(len(unique_uis)), y_ui)):
        tr_uis = set(unique_uis[i] for i in tr_ui_idx)
        val_uis = set(unique_uis[i] for i in val_ui_idx)
        
        tr_idx = [i for i, s in enumerate(train) if s['user_input'] in tr_uis]
        val_idx = [i for i, s in enumerate(train) if s['user_input'] in val_uis]
        
        tr_samp = [train[i] for i in tr_idx]
        val_samp = [train[i] for i in val_idx]
        tr_y = y_all[tr_idx]
        
        tw, tc, tf = make_vectorizers()
        tr_uis_t, tr_fcs = extract_texts(tr_samp)
        val_uis_t, val_fcs = extract_texts(val_samp)
        tr_man = extract_manual(tr_samp)
        val_man = extract_manual(val_samp)
        
        X_tr = build_X(tr_uis_t, tr_fcs, tr_man, tw, tc, tf, fit=True)
        X_val = build_X(val_uis_t, val_fcs, val_man, tw, tc, tf)
        
        clf = make_clf()
        clf.fit(X_tr, tr_y)
        
        preds = clf.predict(X_val)
        proba = clf.predict_proba(X_val)
        
        for ii, vi in enumerate(val_idx):
            all_preds[vi] = preds[ii]
            all_proba[vi] = proba[ii]
        
        fold_f1 = f1_score(y_all[val_idx], preds, average='macro')
        print(f"  Fold {fold+1}: Macro F1 = {fold_f1:.3f}  (val={len(val_idx)} samples, {len(val_uis)} templates)")
    
    macro_f1 = f1_score(y_all, all_preds, average='macro')
    print(f"\n  Template-CV Macro F1: {macro_f1:.3f}")
    print(classification_report(y_all, all_preds, target_names=le.classes_))
    
    # AUC
    from sklearn.metrics import roc_auc_score
    HIGH_RISK = {'data_exfiltration', 'jailbreak', 'prompt_injection', 'tool_abuse'}
    y_bin = np.array([1 if l in HIGH_RISK else 0 for l in labels])
    hr_idx = [i for i, c in enumerate(le.classes_) if c in HIGH_RISK]
    scores = all_proba[:, hr_idx].sum(axis=1)
    auc = roc_auc_score(y_bin, scores)
    print(f"  AUC (high-risk vs rest): {auc:.3f}")
    
    return macro_f1, all_preds, all_proba, le

