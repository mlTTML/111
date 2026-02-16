数据集和算子权重下载链接：
通过网盘分享的文件：videos.tar.gz等4个文件
链接: https://pan.baidu.com/s/1RPKn6q43EN9pyBS8bKTEbw?pwd=afhw 提取码: afhw 
--来自百度网盘超级会员v1的分享
## 目录结构

```
project/
├── README.md
├── train.py   # DPO 训练脚本
├── inference.py              # 推理与评估脚本
└── data/
    ├── train.json            # 训练集
    ├── test.json             # 测试集
    ├── Qualified_video.json  # 合格视频列表（可选）
    ├── CharadesEgo_v1_480.tar.gz
    ├── UCF-101.tar.gz
    └── videos.tar.gz
```

## 环境与依赖


```
pip install -r requirements.txt
```
## 数据准备

1. **解压视频数据**（训练/推理前需解压到 `data/` 下）：

   ```bash
   cd /root/autodl-tmp/project/data
   tar -xzvf CharadesEgo_v1_480.tar.gz
   tar -xzvf UCF-101.tar.gz
   tar -xzvf videos.tar.gz
   ```

2. **数据格式**：`train.json` / `test.json` 为列表，每条为 DPO 格式：

## 训练（DPO）

在项目根目录下运行：

  ```bash
  python train.py
  ```

仅用基座（不加载 LoRA）：加 `--use_base_model_only`。
## 推理
在项目根目录下运行：

  ```bash
  python inference.py
  ```
