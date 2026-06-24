import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
import warnings
warnings.filterwarnings("ignore")



from fix_smiles import try_fix_and_parse

# ---- 配置 ----
MORGAN2_BITS  = 2048
MORGAN3_BITS  = 2048
RDKIT_BITS    = 2048
TORSION_BITS  = 2048
ATOMPAIR_BITS = 2048
AVALON_BITS   = 2048
DATA_DIR = "."
FEAT_DIR = "features"
os.makedirs(FEAT_DIR, exist_ok=True)


def mol_to_morgan2(mol, nBits=MORGAN2_BITS):
    if mol is None:
        return np.zeros(nBits, dtype=np.float32)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nBits)
    fp = gen.GetCountFingerprintAsNumPy(mol)
    return np.log1p(fp.astype(np.float32))


def mol_to_morgan3(mol, nBits=MORGAN3_BITS):
    if mol is None:
        return np.zeros(nBits, dtype=np.float32)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=nBits)
    fp = gen.GetCountFingerprintAsNumPy(mol)
    return np.log1p(fp.astype(np.float32))


def mol_to_maccs(mol):
    if mol is None:
        return np.zeros(167, dtype=np.float32)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(167, dtype=np.float32)
    for bit in fp.GetOnBits():
        if bit < 167:
            arr[bit] = 1.0
    return arr


def mol_to_rdkitfp(mol, nBits=RDKIT_BITS):
    if mol is None:
        return np.zeros(nBits, dtype=np.float32)
    fp = Chem.RDKFingerprint(mol, maxPath=6, fpSize=nBits)
    return np.frombuffer(fp.ToBitString().encode(), dtype='u1').astype(np.float32) - ord('0')


def mol_to_torsion(mol, nBits=TORSION_BITS):
    if mol is None:
        return np.zeros(nBits, dtype=np.float32)
    try:
        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=nBits)
        fp = gen.GetCountFingerprintAsNumPy(mol)
        return np.log1p(fp.astype(np.float32))
    except Exception:
        return np.zeros(nBits, dtype=np.float32)


def mol_to_atompair(mol, nBits=ATOMPAIR_BITS):
    if mol is None:
        return np.zeros(nBits, dtype=np.float32)
    try:
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=nBits)
        fp = gen.GetCountFingerprintAsNumPy(mol)
        return np.log1p(fp.astype(np.float32))
    except Exception:
        return np.zeros(nBits, dtype=np.float32)


def mol_to_avalon(mol, nBits=AVALON_BITS):
    if mol is None:
        return np.zeros(nBits, dtype=np.float32)
    try:
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=nBits)
        return np.frombuffer(fp.ToBitString().encode(), dtype='u1').astype(np.float32) - ord('0')
    except Exception:
        return np.zeros(nBits, dtype=np.float32)


# RDKit描述符列表
EXCLUDED_DESC = {
    'Ipc',  # 数值偏大
    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW',
}

DESC_LIST = [(name, func) for name, func in Descriptors.descList
             if name not in EXCLUDED_DESC]
DESC_NAMES = [name for name, _ in DESC_LIST]
print(f"使用 {len(DESC_NAMES)} 个RDKit描述符")


def mol_to_descriptors(mol):
    if mol is None:
        return np.zeros(len(DESC_LIST), dtype=np.float32)
    vals = []
    for name, func in DESC_LIST:
        try:
            v = func(mol)
            if v is None or np.isnan(v) or np.isinf(v) or abs(v) > 1e9:
                v = 0.0
        except Exception:
            v = 0.0
        vals.append(float(v))
    return np.array(vals, dtype=np.float32)


def compute_all_features(df: pd.DataFrame, split_name: str):
    n = len(df)

    morgan2_arr  = np.zeros((n, MORGAN2_BITS), dtype=np.float32)
    morgan3_arr  = np.zeros((n, MORGAN3_BITS), dtype=np.float32)
    maccs_arr    = np.zeros((n, 167), dtype=np.float32)
    rdkitfp_arr  = np.zeros((n, RDKIT_BITS), dtype=np.float32)
    torsion_arr  = np.zeros((n, TORSION_BITS), dtype=np.float32)
    atompair_arr = np.zeros((n, ATOMPAIR_BITS), dtype=np.float32)
    avalon_arr   = np.zeros((n, AVALON_BITS), dtype=np.float32)
    desc_arr     = np.zeros((n, len(DESC_LIST)), dtype=np.float32)

    invalid_indices = []

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 1000 == 0:
            print(f"  进度: {i}/{n}")
        smi = str(row['smiles'])
        mol, fixed_smi = try_fix_and_parse(smi)

        if mol is None:
            invalid_indices.append(i)

        morgan2_arr[i]  = mol_to_morgan2(mol)
        morgan3_arr[i]  = mol_to_morgan3(mol)
        maccs_arr[i]    = mol_to_maccs(mol)
        rdkitfp_arr[i]  = mol_to_rdkitfp(mol)
        torsion_arr[i]  = mol_to_torsion(mol)
        atompair_arr[i] = mol_to_atompair(mol)
        avalon_arr[i]   = mol_to_avalon(mol)
        desc_arr[i]     = mol_to_descriptors(mol)

    print(f"  完成！无法解析的分子数：{len(invalid_indices)}")
    if invalid_indices:
        ids = df.iloc[invalid_indices]['id'].tolist() if 'id' in df.columns else invalid_indices
        print(f"  无效索引的ID：{ids}")

    # 保存特征为npz
    sp.save_npz(f"{FEAT_DIR}/morgan2_{split_name}.npz", sp.csr_matrix(morgan2_arr))
    sp.save_npz(f"{FEAT_DIR}/morgan3_{split_name}.npz", sp.csr_matrix(morgan3_arr))
    sp.save_npz(f"{FEAT_DIR}/maccs_{split_name}.npz", sp.csr_matrix(maccs_arr))
    sp.save_npz(f"{FEAT_DIR}/rdkitfp_{split_name}.npz", sp.csr_matrix(rdkitfp_arr))
    sp.save_npz(f"{FEAT_DIR}/torsion_{split_name}.npz", sp.csr_matrix(torsion_arr))
    sp.save_npz(f"{FEAT_DIR}/atompair_{split_name}.npz", sp.csr_matrix(atompair_arr))
    sp.save_npz(f"{FEAT_DIR}/avalon_{split_name}.npz", sp.csr_matrix(avalon_arr))
    np.save(f"{FEAT_DIR}/rdkit_desc_{split_name}.npy", desc_arr)

    # 合并为一整包用于树模型的指纹
    all_fps = np.hstack([
        morgan2_arr, morgan3_arr, maccs_arr, rdkitfp_arr,
        torsion_arr, atompair_arr, avalon_arr
    ])
    sp.save_npz(f"{FEAT_DIR}/all_fps_{split_name}.npz", sp.csr_matrix(all_fps))

    print(f"  all_fps shape: {all_fps.shape}")
    print(f"  rdkit_desc shape: {desc_arr.shape}")
    print(f"  已保存至 {FEAT_DIR}/")

    return {
        'morgan2': morgan2_arr,
        'morgan3': morgan3_arr,
        'maccs':   maccs_arr,
        'rdkitfp': rdkitfp_arr,
        'torsion': torsion_arr,
        'atompair': atompair_arr,
        'avalon':  avalon_arr,
        'desc':    desc_arr,
        'all_fps': all_fps,
        'invalid_indices': invalid_indices,
    }


def normalize_descriptors(train_desc, test_desc):
    """使用RobustScaler对描述符进行标准化，并裁剪极值"""
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    train_norm = scaler.fit_transform(train_desc)
    test_norm  = scaler.transform(test_desc)
    # 裁剪极端值
    train_norm = np.clip(train_norm, -5.0, 5.0)
    test_norm  = np.clip(test_norm, -5.0, 5.0)
    
    # 保存scaler以备后用
    import pickle
    with open(f"{FEAT_DIR}/desc_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
        
    np.save(f"{FEAT_DIR}/rdkit_desc_train_norm.npy", train_norm)
    np.save(f"{FEAT_DIR}/rdkit_desc_test_norm.npy",  test_norm)
    print(f"描述符 RobustScaler 标准化完成，保存到 {FEAT_DIR}/")
    return train_norm, test_norm


if __name__ == "__main__":
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df  = pd.read_csv(f"{DATA_DIR}/test.csv")

    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")

    # 提取特征
    train_feats = compute_all_features(train_df, 'train')
    test_feats  = compute_all_features(test_df,  'test')

    # 标准化描述符
    train_desc_norm, test_desc_norm = normalize_descriptors(
        train_feats['desc'], test_feats['desc']
    )

    # 加载预提取hashed FP（已存在于上个阶段的特征提取中）
    hashed_train = sp.load_npz(f"{FEAT_DIR}/hashed_smiles_train.npz")
    hashed_test  = sp.load_npz(f"{FEAT_DIR}/hashed_smiles_test.npz")

    # 合并为全量特征矩阵
    X_train_full = np.hstack([
        train_feats['all_fps'], train_desc_norm, hashed_train.toarray()
    ]).astype(np.float32)
    X_test_full  = np.hstack([
        test_feats['all_fps'],  test_desc_norm,  hashed_test.toarray()
    ]).astype(np.float32)

    np.save(f"{FEAT_DIR}/X_train_full.npy", X_train_full)
    np.save(f"{FEAT_DIR}/X_test_full.npy",  X_test_full)

    print(f"\n{'='*60}")
    print(f"完整特征矩阵已生成:")
    print(f"  X_train_full: {X_train_full.shape}")
    print(f"  X_test_full:  {X_test_full.shape}")
    print(f"{'='*60}")

    # 保存无效信息
    invalid_info = {
        'train_invalid_indices': train_feats['invalid_indices'],
        'test_invalid_indices':  test_feats['invalid_indices'],
        'train_invalid_ids': train_df.iloc[train_feats['invalid_indices']]['id'].tolist() if train_feats['invalid_indices'] else [],
        'test_invalid_ids':  test_df.iloc[test_feats['invalid_indices']]['id'].tolist()   if test_feats['invalid_indices'] else [],
    }
    with open(f"{FEAT_DIR}/invalid_info.json", 'w') as f:
        json.dump(invalid_info, f, indent=2)
    print("\n提取完成")
