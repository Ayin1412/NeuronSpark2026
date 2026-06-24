将src内的内容丢到数据包解压缩根目录
```bash
# 1. 重放演示数据，生成 reconstructed_dataset.npz
python reconstruct_dataset.py

# 2. 训练模型，最佳权重保存到 model/policy.pth
python train.py --epochs 15 --batch-size 256

# 3. 格式校验
python tools/check_format.py .

# 4. 本地评估（360 个任务，批量运行，max_batch_rollouts=64）
python tools/run_public_eval.py . --tasks tasks/valid_tasks.jsonl
```