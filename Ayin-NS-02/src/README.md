```bash
# 1. 安装依赖
pip install -r requirements.txt

# 下载模型文件，可使用modelscope或者hf，也可以直接下载到本地后放在指定目录
# vllm部署，在src运行
./deploy.sh

# 2. 执行主程序
python main.py
```