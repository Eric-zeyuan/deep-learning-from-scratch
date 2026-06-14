# Python 深度学习环境配置

> 本机配置：Windows 11, NVIDIA RTX 4060 Laptop (8GB), Python 3.12 (Microsoft Store)

## 快速恢复环境

```bash
# Python 路径（Microsoft Store 版，已在 PATH 中）
python --version   # Python 3.12.10

# 安装 CUDA PyTorch（从本地 wheel，不用重新下载）
pip install C:\Users\HONOR\dev\torch-wheels\torch-2.6.0+cu124-cp312-cp312-win_amd64.whl
pip install torchvision --index-url https://download.pytorch.org/whl/cu124

# 安装其他常用包
pip install matplotlib numpy tqdm ipython pillow
```

## 验证 GPU

```bash
python -c "import torch; print(torch.cuda.is_available())"  # 必须输出 True
python -c "import torch; print(torch.cuda.get_device_name(0))"  # RTX 4060 Laptop GPU
```

## 关键文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| CUDA torch wheel | `C:\Users\HONOR\dev\torch-wheels\` | 2.5GB, **永久保存，勿删** |
| CIFAR-10 数据集 | `codes/data/cifar-10-batches-py/` | 首次下载 ~170MB |
| Python 包目录 | `%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.12_*\LocalCache\local-packages\Python312\site-packages\` | pip install 的包在这里 |

## 常见坑

1. **num_workers 必须为 0** — Windows 上 DataLoader 用多进程会卡死
2. **pip install torch 默认是 CPU 版** — 必须用 `--index-url https://download.pytorch.org/whl/cu124`
3. **Microsoft Store 版 Python 的包装在奇怪的位置** — 用 `pip list` 确认，不要手动找
4. **pip 缓存会膨胀** — 定期 `pip cache purge` 清理
