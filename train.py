import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from PIL import Image
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import DPOConfig, DPOTrainer
from trl.data_utils import apply_chat_template


def extract_frames(video_path: str, num_frames: int = 12):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    idxs = np.linspace(
        0, total_frames - 1, num=min(num_frames, total_frames), dtype=int
    )
    frames = []
    for frame_idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(image))
    cap.release()
    return frames


def resolve_video_path(path: str) -> str:
    """
    解析视频路径，处理文件名编码不匹配问题。
    文件系统中的文件名可能是Unicode转义形式（如 #U5c4f），而JSON中使用的是原始中文字符。
    """
    import os

    if path.startswith("videos/") or path.startswith("negative_sample/")or path.startswith("CharadesEgo_v1_480/") or path.startswith("UCF-101/"):
        base_path = f"./data/{path}"

        # 首先尝试直接路径
        if os.path.exists(base_path):
            return base_path

        # 如果直接路径不存在，尝试在目录中查找匹配的文件
        # 提取目录和文件名
        if "/" in path:
            dir_part, filename = path.rsplit("/", 1)
            dir_path = f"./data/{dir_part}"
        else:
            dir_path = f"./data/videos"
            filename = path

        if os.path.isdir(dir_path):
            # 在目录中查找匹配的文件
            import re

            # 策略1: 对于negative_sample目录，文件名格式通常是 v_XXX_gYY_cZZ.avi
            # 可以直接匹配文件名（去除扩展名）
            if dir_part == "negative_sample":
                filename_base = os.path.splitext(filename)[0]  # 去除扩展名
                for file in os.listdir(dir_path):
                    file_base = os.path.splitext(file)[0]
                    if filename_base == file_base and (file.endswith('.mp4') or file.endswith('.avi')):
                        matched_path = os.path.join(dir_path, file)
                        if os.path.exists(matched_path):
                            return matched_path

            # 策略2: 匹配日期时间格式（支持多种格式）
            # 格式1: 2025-11-05 16-01-47 或 2025-11-05 16:01:47
            date_time_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}[-:]\d{2}[-:]\d{2})', filename)
            if date_time_match:
                date_time = date_time_match.group(1)
                # 统一格式（将冒号替换为连字符）
                date_time_normalized = date_time.replace(':', '-')
                # 查找包含该日期时间的文件
                for file in os.listdir(dir_path):
                    file_normalized = file.replace(':', '-')
                    if date_time_normalized in file_normalized and (file.endswith('.mp4') or file.endswith('.avi')):
                        matched_path = os.path.join(dir_path, file)
                        if os.path.exists(matched_path):
                            return matched_path

            # 策略3: 提取完整时间戳格式：2025-10-22 093642（旧格式）
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{6})', filename)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                # 查找包含该时间戳的文件
                for file in os.listdir(dir_path):
                    if timestamp in file and (file.endswith('.mp4') or file.endswith('.avi')):
                        matched_path = os.path.join(dir_path, file)
                        if os.path.exists(matched_path):
                            return matched_path

            # 策略4: 如果完整时间戳匹配失败，尝试只匹配6位数字时间戳（如 093642）
            timestamp_match = re.search(r'(\d{6})', filename)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                # 查找包含该时间戳的文件
                for file in os.listdir(dir_path):
                    if timestamp in file and (file.endswith('.mp4') or file.endswith('.avi')):
                        matched_path = os.path.join(dir_path, file)
                        if os.path.exists(matched_path):
                            return matched_path

            # 策略5: 尝试匹配日期部分（YYYY-MM-DD）并配合文件名主要部分
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
            if date_match:
                date = date_match.group(1)
                # 提取文件名的主要部分（去除日期后的部分，用于进一步匹配）
                main_part = filename.split(date)[0].strip()
                if main_part and len(main_part) > 5:  # 至少5个字符才匹配
                    main_part_clean = re.sub(r'[^\w\s-]', '', main_part)[:30]  # 只保留字母数字，取前30个字符
                    for file in os.listdir(dir_path):
                        if date in file:
                            file_clean = re.sub(r'[^\w\s-]', '', file)[:30]
                            if main_part_clean in file_clean or file_clean in main_part_clean:
                                if file.endswith('.mp4') or file.endswith('.avi'):
                                    matched_path = os.path.join(dir_path, file)
                                    if os.path.exists(matched_path):
                                        return matched_path

        # 如果都找不到，返回原始路径（让后续处理报错）
        return base_path

    return path


def load_video_pref_dataset(
        json_path: Path, frames_per_video: int = 12, return_kept_entries: bool = False
) -> Dataset:
    """
    加载视频偏好数据集，符合TRL DPO对话格式要求。
    根据官方文档，prompt/chosen/rejected都应该是消息列表格式。
    """
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    kept_entries = [] if return_kept_entries else None
    for entry in raw:
        # 直接使用JSON中的prompt消息列表（已经是正确的对话格式）
        prompt_messages = entry.get("prompt", [])
        if not prompt_messages:
            continue

        # 提取视频帧并转换video块为image块
        # 根据TRL视觉数据集格式要求，content中应该使用{"type": "image"}而不是{"type": "video"}
        frames = []
        processed_prompt_messages = []

        for turn in prompt_messages:
            processed_turn = turn.copy()
            if turn.get("role") == "user":
                processed_content = []
                image_counter = 0  # 用于跟踪当前视频对应的图像索引

                for block in turn.get("content", []):
                    if block.get("type") == "video":
                        # 提取视频帧
                        abs_path = resolve_video_path(block["path"])
                        video_frames = extract_frames(abs_path, frames_per_video)

                        # 检查是否成功提取帧
                        if not video_frames:
                            print(f"Warning: Failed to extract frames from {abs_path}, skipping this video block")
                            continue  # 跳过这个video块，不添加image块

                        frames.extend(video_frames)

                        # 将video块转换为对应数量的image块
                        # 根据TRL视觉数据集格式要求：
                        # - image块格式：{"type": "image"}（不需要text字段）
                        # - text块格式：{"type": "text", "text": "..."}
                        for _ in range(len(video_frames)):
                            processed_content.append({"type": "image"})  # 符合官方格式：只有type字段
                            image_counter += 1
                    else:
                        # 保留非video块（如text块）
                        processed_content.append(block)

                processed_turn["content"] = processed_content

            processed_prompt_messages.append(processed_turn)

        # 确保images列表不为空，且与prompt中的image块数量匹配
        # 如果frames为空，说明没有成功提取任何视频帧，跳过这个样本
        if not frames:
            print(f"Warning: No frames extracted for entry, skipping")
            continue

        # 验证images数量与prompt中image块数量匹配（符合TRL官方要求）
        # 统计prompt中所有image块的数量
        image_block_count = 0
        for turn in processed_prompt_messages:
            if "content" in turn:
                for block in turn.get("content", []):
                    if block.get("type") == "image":
                        image_block_count += 1

        if len(frames) != image_block_count:
            print(
                f"Warning: Images count ({len(frames)}) doesn't match "
                f"image blocks count ({image_block_count}) in prompt. Skipping."
            )
            continue

        # 清理prompt中的image块，确保符合TRL格式要求
        # 根据官方文档，image块应该只有{"type": "image"}，不应该有text字段
        cleaned_prompt = []
        for turn in processed_prompt_messages:
            cleaned_turn = turn.copy()
            if "content" in cleaned_turn:
                cleaned_content = []
                for block in cleaned_turn["content"]:
                    if block.get("type") == "image":
                        # 确保image块只有type字段，移除任何text字段
                        cleaned_content.append({"type": "image"})
                    else:
                        # 保留其他类型的块（如text块）
                        cleaned_content.append(block)
                cleaned_turn["content"] = cleaned_content
            cleaned_prompt.append(cleaned_turn)

        # 根据TRL DPO要求，使用转换后的消息列表格式
        # prompt中的video块已转换为image块，符合TRL视觉数据集格式
        # 根据官方文档：https://hugging-face.cn/docs/trl/dataset_formats
        # - prompt/chosen/rejected都应该是消息列表格式
        # - images字段包含PIL.Image对象列表，顺序与prompt中image块顺序一致
        samples.append(
            {
                "prompt": cleaned_prompt,  # 已清理的消息列表格式（image块只有type字段）
                "images": frames,  # 所有视频帧的列表（PIL.Image对象），顺序与prompt中image块一致
                "chosen": entry["chosen"],  # 消息列表格式
                "rejected": entry["rejected"],  # 消息列表格式
            }
        )
        if kept_entries is not None:
            kept_entries.append(entry)

    dataset = Dataset.from_list(samples)
    if kept_entries is not None:
        return dataset, kept_entries
    return dataset


class TensorBoardCallback(TrainerCallback):
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
        self.writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL LoRA DPO fine-tuning on video_pref_train_10.json"
    )
    parser.add_argument(
        "--base_model",
        default="./Qwen3-VL-8B-Instruct",
        help="Qwen3-VL 基座路径",
    )
    parser.add_argument(
        "--data_file",
        default="./data/train.json",
        help="偏好数据 JSON 路径",
    )
    parser.add_argument(
        "--output_dir",
        default="./qwen3vl-lora-video-pref",
        help="LoRA 结果输出目录",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,  # 从1增加到4，使用更大的有效batch size，提高训练稳定性
        help="梯度累积步数，增加有效batch size",
    )
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数（当未指定 --max_steps 时生效）")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="最大训练步数；若指定（如 500），则按步数训练并覆盖 --epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,  # 从5e-6降低到1e-6，提供更稳定的训练
        help="DPO训练推荐使用较小的学习率（1e-6到5e-6）",
    )
    parser.add_argument("--max_len", type=int, default=25600)
    parser.add_argument("--frames_per_video", type=int, default=8)
    parser.add_argument(
        "--adapter_checkpoint",
        type=str,
        default=None,
        help="SFT阶段生成的LoRA权重路径，若提供则以其作为策略初始化与参考模型",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,  # 从0.1增加到0.3，提供更强的KL散度约束，防止训练不稳定
        help="DPO beta参数，控制KL散度权重，推荐0.1-0.5。目标KL散度应该保持在0到10之间（参考TRL文档）",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="梯度裁剪阈值，防止梯度爆炸",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="每多少步保存一次权重（默认 5）",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        help="TensorBoard 日志目录（如果未指定，将自动生成）",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="TensorBoard运行名称（用于区分不同训练，如果未指定将自动生成）",
    )
    parser.add_argument(
        "--debug_samples",
        type=int,
        default=0,
        help="打印前 N 条格式化样本供调试",
    )
    parser.add_argument("--debug_token_stats", action="store_true",
                        help="统计每个样本input_ids总长度分布，仅调试阶段分析用")
    args = parser.parse_args()

    loader_result = load_video_pref_dataset(
        Path(args.data_file),
        frames_per_video=args.frames_per_video,
        return_kept_entries=args.debug_token_stats,
    )
    if args.debug_token_stats:
        dataset, raw_entries = loader_result
    else:
        dataset = loader_result
        raw_entries = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=False,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    if args.debug_token_stats:
        print("\n统计全部训练样本 input_ids token 总长度 (文本+图片):\n")
        lengths = []
        filtered_entries = []
        for idx, sample in enumerate(dataset):
            prompt = sample["prompt"]
            images = sample.get("images", [])
            prompt_text = apply_chat_template({"prompt": prompt}, tokenizer)["prompt"]
            inputs = processor(
                images=images,
                text=prompt_text,
                return_tensors='pt'  # 关键，强制返回tensor
            )
            input_ids = inputs["input_ids"]
            # 判断 input_ids shape，自动统计真实token总长
            if hasattr(input_ids, 'shape') and input_ids.ndim == 2:
                curlen = input_ids.shape[1]
            elif hasattr(input_ids, 'shape') and input_ids.ndim == 1:
                curlen = input_ids.shape[0]
            else:
                curlen = len(input_ids)
            lengths.append(curlen)
            print(f"样本{idx + 1:4d}: input_ids总长度 = {curlen}")
            if curlen <= args.max_len:
                if raw_entries is None or idx >= len(raw_entries):
                    print(
                        "  [警告] 无法找到对应的原始样本，跳过写入过滤结果。"
                    )
                else:
                    filtered_entries.append(raw_entries[idx])
            if curlen == 1:
                print(f"  [警告] 该样本得到的token只有1，可能未正确处理图片与文本，inputs结构如下：\n{inputs}\n")
        if lengths:
            import numpy as np
            arr = np.array(lengths)
            print("\n--- input_ids长度统计结果 ---")
            print(f"最大长度: {arr.max()}\n最小长度: {arr.min()}\n平均长度: {arr.mean():.2f}")
            print("建议max_len设置 >= 最大token长度，且结合显存实际做权衡。\n")
            if filtered_entries:
                filtered_path = Path(args.data_file).with_name(
                    f"{Path(args.data_file).stem}_filtered_maxlen{args.max_len}.json"
                )
                filtered_path.write_text(
                    json.dumps(filtered_entries, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"已保存 {len(filtered_entries)} 条满足 max_len <= {args.max_len} 的样本到 {filtered_path}"
                )
            else:
                print(f"没有样本满足 max_len <= {args.max_len} 的条件，未生成过滤文件。")
        else:
            print("数据样本为空，未统计。\n")
        import sys
        sys.exit(0)

    if tokenizer.chat_template is None:
        tmpl_path = Path(args.base_model) / "chat_template.json"
        if tmpl_path.exists():
            tokenizer.chat_template = tmpl_path.read_text()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # 加载基础模型（用于DPO训练）
    policy_base_model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    # 配置LoRA适配器
    default_lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    reference_model = None
    peft_config_for_trl = default_lora_config

    if args.adapter_checkpoint:
        # 官方建议：先做SFT以得到同分布策略；该LoRA权重既是DPO初始化，也是ref
        PeftConfig.from_pretrained(args.adapter_checkpoint)
        model = PeftModel.from_pretrained(
            policy_base_model,
            args.adapter_checkpoint,
            is_trainable=True,
        )

        ref_base_model = AutoModelForVision2Seq.from_pretrained(
            args.base_model,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        reference_model = PeftModel.from_pretrained(
            ref_base_model,
            args.adapter_checkpoint,
            is_trainable=False,
        )
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
        peft_config_for_trl = None
    else:
        model = get_peft_model(policy_base_model, default_lora_config)

    # 数据集已经在 load_video_pref_dataset 中格式化完成，不需要再次 map
    # 如果设置了 debug_samples，打印调试信息
    if args.debug_samples > 0:
        print(f"\n====== 调试前 {args.debug_samples} 个样本 ======")
        for idx in range(min(args.debug_samples, len(dataset))):
            example = dataset[idx]
            print(f"\n样本 {idx + 1}:")
            print("Prompt messages:", example["prompt"])
            print("Chosen messages:", example["chosen"])
            print("Rejected messages:", example["rejected"])
            print("Total frames:", len(example.get("images", [])))
        print("====== 调试结束 ======\n")

    formatted_dataset = dataset

    # 根据TRL官方文档（https://hugging-face.cn/docs/trl/how_to_train）的建议：
    # 1. beta参数：控制KL散度权重，防止模型过度偏离参考模型
    #    - 推荐范围：0.1-0.5
    #    - 目标KL散度应该保持在0到10之间
    # 2. 学习率：DPO训练推荐使用较小的学习率（5e-6到1e-5）
    # 3. 梯度裁剪：防止梯度爆炸，推荐max_grad_norm=1.0
    # 4. 监控指标：应该关注rewards/margins和rewards/accuracies，而不仅仅是loss
    #    - rewards/margins：应该为正值（chosen奖励 > rejected奖励）
    #    - rewards/accuracies：应该接近1.0（模型正确选择chosen）
    training_conf = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        max_length=args.max_len,
        learning_rate=args.learning_rate,
        beta=args.beta,  # DPO关键参数：控制KL散度权重，防止模型过度偏离参考模型
        max_grad_norm=args.max_grad_norm,  # 梯度裁剪，防止梯度爆炸
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.2,  # 增加warmup比例，让训练更平稳（从0.1增加到0.2）
        bf16=torch.cuda.is_available(),
        # 注意：根据TRL文档，在RL训练中，损失不是主要指标
        # 应该关注rewards/margins（应该为正值）和rewards/accuracies（应该接近1.0）
    )

    # 如果使用SFT得到的LoRA权重，则参考模型固定为冻结的SFT策略；
    # 否则退化为默认LoRA配置并依赖TRL在内部构建参考策略。
    # 自动生成TensorBoard运行名称和日志目录
    if args.run_name is None:
        # 根据关键超参数生成描述性的运行名称
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"lr{args.learning_rate}_beta{args.beta}_grad{args.max_grad_norm}_{timestamp}"

    if args.log_dir is None:
        # 使用run_name创建日志目录
        args.log_dir = f"./runs/{args.run_name}"

    # 确保日志目录存在
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    print(f"TensorBoard运行名称: {args.run_name}")
    print(f"TensorBoard日志目录: {args.log_dir}")
    print(f"提示: 使用 'tensorboard --logdir ./runs' 查看所有训练的对比\n")

    trainer = DPOTrainer(
        model=model,  # 带LoRA适配器的训练模型
        ref_model=reference_model,  # 若提供SFT权重，则显式使用其作为参考策略
        args=training_conf,
        train_dataset=formatted_dataset,
        processing_class=processor,  # 对于VLM模型，根据官方文档和示例，应该使用processor而不是tokenizer
        peft_config=peft_config_for_trl,  # 仅在未提供SFT权重时让TRL自动创建参考模型
    )
    trainer.add_callback(TensorBoardCallback(args.log_dir))

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

