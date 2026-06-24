## 1. 代码文件一览

- **[fix_smiles.py](fix_smiles.py)**：SMILES 修复工具。用于解决 RDKit 解析部分畸变/超价分子会崩溃的问题，通过正则替换和宽松加载还原合法的分子结构。
- **[build_features_v2.py](build_features_v2.py)**：特征生成脚本。提取 Morgan、Torsion、Atom Pairs 计数指纹并做 `log1p` 平滑，同时计算 2D 物理描述符并用 RobustScaler 缩放。
- **[train_model_full.py](train_model_full.py)**：基模型训练脚本。提取 Murcko Scaffold 骨架做 5 折 GroupKFold 分组，训练 LightGBM、XGBoost、CatBoost 和 MLP 并输出 OOF 预测。
- **[train_stacking_v2.py](train_stacking_v2.py)**：元回归融合脚本。提取 4 个基模型的 OOF 统计元特征，并拼入相关性前 50 的 2D 物理描述符，使用 BayesianRidge 进行 Stacking 训练与推理。
- **[predict_and_submit_full_opt.py](predict_and_submit_full_opt.py)**：最终提交生成脚本。加载 Stacking 的预测结果，进行格式和极值自检后输出 `results.csv`。
- **[requirements.txt](requirements.txt)**：Python 依赖包列表。

---


在运行代码前，请确保将官方的 `train.csv` 和 `test.csv` 放在本目录下

安装所需的依赖包：
```bash
pip install -r requirements.txt
```

---


### 特征计算与异常分子修复
```bash
python build_features_v2.py
```
### 训练 5 折基模型并计算 OOF
```bash
python train_model_full.py
```

### 运行元回归 Stacking 融合
```bash
python train_stacking_v2.py
```

### 生成最终结果
```bash
python predict_and_submit_full_opt.py
```
