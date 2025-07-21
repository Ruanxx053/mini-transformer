# Mini-Transformer from Scratch
> 从零手写一个可训练的 Transformer（不依赖高层封装），在 IWSLT14 De-En 复现 Base BLEU 34.8。

## 📊 Benchmark
| Model           | Dataset  | Metric | Our Score | Original |
|-----------------|----------|--------|-----------|----------|
| Mini-Transformer| IWSLT14  | BLEU   | 34.8      | 34.6     |

## 🚀 Quick Start (Windows)
```powershell
pip install -r requirements.txt
python scripts\train.py --config config\base.yaml