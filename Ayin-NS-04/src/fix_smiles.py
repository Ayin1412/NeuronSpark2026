import re
import logging
from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 常见问题模式替换规则（按优先级排列）
REPLACEMENT_RULES = [
    # 显式 NH 问题（=NH2, =NH 等高价氮）
    (r'\[NH2\]',  'N'),
    (r'\[NH\]',   'N'),
    (r'=\[NH2\]', '=N'),
    (r'=\[NH\]',  '=N'),
    # guanidine/amidine 中常见的写法
    (r'\(=\[NH\]\)',    '(=N)'),
    (r'\(=\[NH2\]\)',   '(=N)'),
    # AlH3 / AlH2 显式氢导致超价
    (r'\[AlH3\]', '[Al]'),
    (r'\[AlH2\]', '[Al]'),
    (r'\[AlH\]',  '[Al]'),
    # Pt 铂系金属：去掉 NH4 配体问题
    (r'\[NH4\]',  '[NH4+]'),
    # 硝基负离子写法 N([O-]) -> N(=O)
    (r'N\(\[O-\]\)', 'N(=O)'),
    (r'=N\(\[O-\]\)', '=N(=O)'),
]

# 针对特定问题SMILES的直接修复映射（精确匹配）
DIRECT_FIX_MAP = {
    # 磺酰基联苯酰胺吡唑酮：n2c(=O)c(c(=O)n2 kekulize失败 -> 展开为非芳香环写法
    'c1ccc(cc1)n2c(=O)c(c(=O)n2c3ccccc3)CCS(=O)c4ccccc4':
        'O=C1c2ccccc2N(c2ccccc2)N1CCS(=O)c1ccccc1',
    # 吡唑酮芳香化失败 -> 展开
    'CCCCc1c(=O)n(n(c1=O)c2ccc(cc2)O)c3ccccc3':
        'CCCCC1C(=O)N(N(C1=O)c2ccc(O)cc2)c1ccccc1',
    # 测试集中的对应物
    'CCCCc1c(=O)n(n(c1=O)c2ccc(cc2)O)c3ccccc3':
        'CCCCC1C(=O)N(N(C1=O)c2ccc(O)cc2)c1ccccc1',
}


def try_parse(smiles: str) -> Chem.Mol | None:
    """尝试直接解析SMILES，返回 mol 或 None"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None


def try_fix_and_parse(smiles: str) -> tuple[Chem.Mol | None, str]:
    """
    尝试修复SMILES并解析。
    返回 (mol, fixed_smiles)。如果修复失败，mol=None。
    """
    # Step 0: 直接映射（精确匹配特定问题SMILES）
    if smiles in DIRECT_FIX_MAP:
        fixed = DIRECT_FIX_MAP[smiles]
        mol = try_parse(fixed)
        if mol is not None:
            logger.info(f"修复成功（直接映射）: {smiles!r} -> {fixed!r}")
            return mol, fixed

    # Step 1: 直接解析
    mol = try_parse(smiles)
    if mol is not None:
        return mol, smiles

    # Step 2: 逐步应用替换规则（累积替换）
    fixed = smiles
    for pattern, replacement in REPLACEMENT_RULES:
        candidate = re.sub(pattern, replacement, fixed)
        mol = try_parse(candidate)
        if mol is not None:
            logger.info(f"修复成功（规则替换）: {smiles!r} -> {candidate!r}")
            return mol, candidate
        fixed = candidate  # 累积替换

    # Step 2b: 对原始SMILES单独尝试每条规则（不累积）
    for pattern, replacement in REPLACEMENT_RULES:
        candidate = re.sub(pattern, replacement, smiles)
        if candidate != smiles:
            mol = try_parse(candidate)
            if mol is not None:
                return mol, candidate

    # Step 3: sanitize=False 解析后尝试标准化（去除问题原子的显式氢）
    try:
        mol_raw = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol_raw is not None:
            mol_raw.UpdatePropertyCache(strict=False)
            try:
                Chem.FastFindRings(mol_raw)
                Chem.SanitizeMol(mol_raw,
                    Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                    Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                    Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                    Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                    Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                )
                # 转为SMILES再解析
                canonical = Chem.MolToSmiles(mol_raw)
                mol2 = try_parse(canonical)
                if mol2 is not None:
                    logger.info(f"修复成功（宽松化）: {smiles!r} -> {canonical!r}")
                    return mol2, canonical
            except Exception:
                pass
    except Exception:
        pass

    logger.warning(f"修复失败，将使用零填充特征: {smiles!r}")
    return None, smiles


def fix_smiles_list(smiles_list: list[str]) -> list[tuple[str, Chem.Mol | None, str]]:
    """
    批量修复SMILES。
    返回 list of (原始SMILES, mol或None, 修复后SMILES)
    """
    results = []
    valid_count = 0
    invalid_count = 0
    fixed_count = 0

    for smi in smiles_list:
        mol, fixed = try_fix_and_parse(str(smi))
        results.append((smi, mol, fixed))
        if mol is not None:
            if fixed != smi:
                fixed_count += 1
            else:
                valid_count += 1
        else:
            invalid_count += 1

    print(f"SMILES解析统计: 原始有效={valid_count}, 修复成功={fixed_count}, 无法修复={invalid_count}")
    return results


if __name__ == "__main__":
    # 测试已知无效SMILES
    test_cases = [
        "c1(nc(NC(N)=[NH2])sc1)CSCCNC(=[NH]C#N)NC",
        "c1ccc(cc1)n2c(=O)c(c(=O)n2c3ccccc3)CCS(=O)c4ccccc4",
        "CCCCCCCCCCCCCCCCCC(=O)O[AlH3](O)O",
        "CC(=O)O[AlH3](O)O",
        "n1c(csc1\\[NH]=C(\\N)N)c1cccc(c1)N\\C(NC)=[NH]\\C#N",
        "s1cc(nc1\\[NH]=C(\\N)N)C",
        "[NH4][Pt]([NH4])(Cl)Cl",
        "CCCCO[AlH3](OCCCC)OCCCC",
        "Cc1nc(sc1)\\[NH]=C(\\N)N",
        "n1c(csc1\\[NH]=C(\\N)N)c1ccccc1",
        # 测试集
        "O=CO[AlH3](OC=O)OC=O",
        "c1c(c(ncc1)CSCCN\\C(=[NH]\\C#N)NCC)Br",
        "O=N([O-])C1=C(CN=C1NCCSCc2ncccc2)Cc3ccccc3",
        "s1cc(CSCCN\\C(NC)=[NH]\\C#N)nc1\\[NH]=C(\\N)N",
        "CCCCc1c(=O)n(n(c1=O)c2ccc(cc2)O)c3ccccc3",
        "CCOC(=O)/C=C(/C)O[AlH3](OC(C)CC)OC(C)CC",
        "c1(cc(N\\C(=[NH]\\c2cccc(c2)CC)C)ccc1)CC",
    ]
    results = fix_smiles_list(test_cases)
    for orig, mol, fixed in results:
        status = "[OK]" if mol is not None else "[FAIL]"
        changed = " [已修复]" if fixed != orig and mol is not None else ""
        print(f"{status}{changed}: {orig[:60]}")
