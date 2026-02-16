import argparse
import copy
import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set

import torch
from PIL import Image
import cv2
import numpy as np
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from trl.data_utils import apply_chat_template


def extract_frames(video_path: str, num_frames: int = 12):
    """从视频中提取帧"""
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
    import re

    if path.startswith("videos/") or path.startswith(
            "CharadesEgo_v1_480/") or path.startswith("UCF-101/"):
        base_path = f"/root/autodl-tmp/common_benchmark/{path}"

        # 首先尝试直接路径
        if os.path.exists(base_path):
            return base_path

        # 如果直接路径不存在，尝试在目录中查找匹配的文件
        # 提取目录和文件名
        if "/" in path:
            dir_part, filename = path.rsplit("/", 1)
            dir_path = f"/root/autodl-tmp/common_benchmark/{dir_part}"
        else:
            dir_path = f"/root/autodl-tmp/common_benchmark/videos"
            filename = path

        if os.path.isdir(dir_path):
            # 策略1: 对于negative_sample目录，文件名格式通常是 v_XXX_gYY_cZZ.avi
            # 可以直接匹配文件名（去除扩展名）
            if dir_part == "negative_sample":
                filename_base = os.path.splitext(filename)[0]
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
                date_time_normalized = date_time.replace(':', '-')
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
                for file in os.listdir(dir_path):
                    if timestamp in file and (file.endswith('.mp4') or file.endswith('.avi')):
                        matched_path = os.path.join(dir_path, file)
                        if os.path.exists(matched_path):
                            return matched_path

            # 策略4: 如果完整时间戳匹配失败，尝试只匹配6位数字时间戳（如 093642）
            timestamp_match = re.search(r'(\d{6})', filename)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                for file in os.listdir(dir_path):
                    if timestamp in file and (file.endswith('.mp4') or file.endswith('.avi')):
                        matched_path = os.path.join(dir_path, file)
                        if os.path.exists(matched_path):
                            return matched_path

            # 策略5: 尝试匹配日期部分（YYYY-MM-DD）并配合文件名主要部分
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
            if date_match:
                date = date_match.group(1)
                main_part = filename.split(date)[0].strip()
                if main_part and len(main_part) > 5:
                    main_part_clean = re.sub(r'[^\w\s-]', '', main_part)[:30]
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


def extract_answer(text: str) -> str:
    """从模型输出中提取 yes 或 no"""
    text = text.lower().strip()
    # 移除标点符号和空格，只保留字母
    text_clean = ''.join(c for c in text if c.isalpha())

    # 检查是否包含 yes 或 no
    if "yes" in text_clean:
        return "yes"
    elif "no" in text_clean:
        return "no"
    else:
        # 如果都没有，尝试查找 "yes" 或 "no" 作为完整单词
        words = text.split()
        for word in words:
            word_clean = ''.join(c for c in word.lower() if c.isalpha())
            if word_clean == "yes":
                return "yes"
            elif word_clean == "no":
                return "no"
        # 如果还是找不到，返回 "unknown"
        return "unknown"


def get_chosen_answer(chosen: List[Dict]) -> str:
    """从 chosen 消息中提取正确答案"""
    for msg in chosen:
        if msg.get("role") == "assistant":
            for content in msg.get("content", []):
                if content.get("type") == "text":
                    text = content.get("text", "").lower().strip()
                    if "yes" in text:
                        return "yes"
                    elif "no" in text:
                        return "no"
    return "unknown"


def compute_roc_auc(labels: List[int], scores: List[float]) -> float:
    """
    简单实现 ROC AUC 计算（不依赖 sklearn），返回 0-100 之间的百分比。
    labels: 0/1，1 表示正样本（yes）
    scores: 任意实数，越大表示越偏向正样本
    """
    n = len(labels)
    if n == 0:
        return 0.0
    pos = sum(labels)
    neg = n - pos
    if pos == 0 or neg == 0:
        # 只有单一类别，ROC AUC 没有意义
        return 0.0

    # 按 score 从小到大排序
    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    tp = fp = 0
    prev_tpr = prev_fpr = 0.0
    auc = 0.0

    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / pos
        fpr = fp / neg
        # 梯形面积
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_tpr, prev_fpr = tpr, fpr

    return auc * 100.0


def compute_p_at_k(labels: List[int], scores: List[float], k: int) -> Tuple[float, int]:
    """
    计算 Top-K Precision:
    - labels: 0/1，1 表示正样本（yes）
    - scores: 实数分数，越大越偏向正样本
    - k: 期望的 K 值
    返回 (P@K 百分比, 实际使用的 K)
    """
    n = len(labels)
    if n == 0 or k <= 0:
        return 0.0, 0
    k = min(k, n)
    sorted_idx = sorted(range(n), key=lambda i: scores[i], reverse=True)
    top_indices = sorted_idx[:k]
    hits = sum(labels[i] for i in top_indices)
    return hits / k * 100.0, k


def run_inference_for_checkpoint(
        model,
        processor,
        tokenizer,
        test_data: List[Dict],
        checkpoint_name: str,
        frames_per_video: int,
        max_new_tokens: int,
        max_length: int = 25600,  # 超出该长度的样本将被跳过
        top_k: int = 1000,  # 用于计算 P@K 的 K，若 <=0 则不计算
        debug_token_stats: bool = False,
        debug_dataset: Optional[List[Dict]] = None,
        debug_hashes: Optional[Set[str]] = None,
) -> Tuple[List[Dict], Dict[str, float]]:
    """对单个 checkpoint 运行推理，返回结果和统计信息"""
    print(f"\n{'=' * 60}")
    print(f"开始推理: {checkpoint_name}")
    print(f"{'=' * 60}")

    results = []
    skipped_samples = 0
    cuda_error_samples = 0
    correct_yes = 0
    total_yes = 0
    correct_no = 0
    total_no = 0
    predicted_yes = 0  # 模型预测为yes的总数（用于计算查准率）

    # 用于 AUC / P@K 的打分与标签
    auc_labels: List[int] = []  # 1=gold yes, 0=gold no
    auc_scores: List[float] = []  # 越大表示越偏向 yes

    for idx, entry in enumerate(test_data):
        if (idx + 1) % 10 == 0:
            print(f"处理进度: {idx + 1}/{len(test_data)}")

        # 提取视频帧并准备消息格式（与训练时一致）
        prompt_messages = entry.get("prompt", [])
        video_frames_list = []

        # 先提取所有视频帧
        for turn in prompt_messages:
            if turn.get("role") == "user":
                for block in turn.get("content", []):
                    if block.get("type") == "video":
                        abs_path = resolve_video_path(block["path"])
                        video_frames = extract_frames(abs_path, frames_per_video)
                        if not video_frames:
                            print(f"Warning: 无法从 {abs_path} 提取帧，跳过样本 {idx}")
                            break
                        video_frames_list.append(video_frames)

        if not video_frames_list:
            print(f"Warning: 样本 {idx} 没有提取到帧，跳过")
            continue

        # 准备消息格式：与训练时保持一致
        processed_prompt_messages = []
        all_frames = []

        for turn in prompt_messages:
            processed_turn = turn.copy()
            if turn.get("role") == "user":
                processed_content = []
                video_idx = 0

                for block in turn.get("content", []):
                    if block.get("type") == "video":
                        if video_idx < len(video_frames_list):
                            video_frames = video_frames_list[video_idx]
                            all_frames.extend(video_frames)

                            # 添加 image 块，格式与训练时一致：只有 {"type": "image"}
                            for _ in range(len(video_frames)):
                                processed_content.append({"type": "image"})
                            video_idx += 1
                        else:
                            print(f"Warning: 样本 {idx} 的 video 块索引超出范围")
                    else:
                        processed_content.append(block)
                processed_turn["content"] = processed_content
            processed_prompt_messages.append(processed_turn)

        # 使用与训练时一致的 prompt 处理方式
        skip_for_stats = False
        try:
            torch.cuda.empty_cache()

            # 使用 apply_chat_template（与训练时一致）
            prompt_text = apply_chat_template({"prompt": processed_prompt_messages}, tokenizer)["prompt"]

            # 使用 processor（与训练时一致）
            # 注意：processor 可能不支持 max_length 参数，需要根据实际情况调整
            inputs = processor(
                images=all_frames,
                text=prompt_text,
                return_tensors='pt',
                padding=True,
            )

            # 清理中间变量
            del processed_prompt_messages, all_frames, prompt_text
            torch.cuda.empty_cache()

            # 在搬运到 GPU 前检查文本长度，必要时记录样本
            length_threshold = max_length if max_length is not None else 25600
            input_ids_check = inputs.get("input_ids", None) if isinstance(inputs, dict) else None
            actual_length = None
            if input_ids_check is not None:
                actual_length = input_ids_check.shape[1] if input_ids_check.ndim == 2 else input_ids_check.shape[0]

            if debug_token_stats and actual_length is not None and length_threshold is not None:
                if actual_length <= length_threshold and debug_dataset is not None:
                    sample_key = json.dumps(entry, ensure_ascii=False, sort_keys=True)
                    if debug_hashes is None or sample_key not in debug_hashes:
                        debug_dataset.append(copy.deepcopy(entry))
                        if debug_hashes is not None:
                            debug_hashes.add(sample_key)

            if actual_length is not None and max_length is not None and actual_length > max_length:
                print(
                    f"Skip: 样本 {idx} 的输入长度 {actual_length} 超过 max_len {max_length}，跳过推理"
                )
                skipped_samples += 1
                # 清理当前样本占用的资源后跳过
                del inputs, input_ids_check
                if 'video_frames_list' in locals():
                    for frames in video_frames_list:
                        for frame in frames:
                            del frame
                        del frames
                    del video_frames_list
                torch.cuda.empty_cache()
                continue

            # 将输入移动到模型设备
            if isinstance(inputs, dict):
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                          for k, v in inputs.items()}
            else:
                inputs = inputs.to(model.device)

            with torch.no_grad():
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else None,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            generated_ids = gen_out.sequences
            gen_scores = gen_out.scores  # List[Tensor]，每一步的 logits

            # 解码输出
            input_ids = inputs["input_ids"]
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            # 计算生成序列的 log 概率，作为置信度（越大越自信）
            seq_logprob = 0.0
            try:
                if isinstance(gen_scores, (list, tuple)) and len(gen_scores) > 0:
                    # 目前 batch_size=1，因此只取第一个样本
                    gen_tokens = generated_ids_trimmed[0]
                    # gen_scores 的长度应与生成步数一致，如有不一致取两者较小长度
                    steps = min(len(gen_scores), len(gen_tokens))
                    logprobs = []
                    for t in range(steps):
                        logits = gen_scores[t][0]  # (vocab_size,)
                        logp = torch.log_softmax(logits, dim=-1)
                        token_id = gen_tokens[t]
                        logprobs.append(logp[token_id])
                    if logprobs:
                        seq_logprob = float(torch.stack(logprobs).sum().cpu())
            except Exception:
                # 置信度计算失败时，不影响主流程
                seq_logprob = 0.0

            predicted_answer = extract_answer(output_text)

            # 清理
            del inputs, input_ids, generated_ids, generated_ids_trimmed, gen_out
            torch.cuda.empty_cache()

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_msg = str(e)
            if "CUDA" in error_msg or "cuda" in error_msg or "illegal memory" in error_msg.lower():
                print(f"CUDA Error processing sample {idx}: {error_msg}")
                predicted_answer = "error_cuda"
                output_text = f"Error: CUDA error - {error_msg[:100]}"  # 截断错误信息
                skip_for_stats = True
                cuda_error_samples += 1
            else:
                print(f"Runtime Error processing sample {idx}: {error_msg}")
                predicted_answer = "error_runtime"
                output_text = f"Error: Runtime error - {error_msg[:100]}"
            # 尝试清理 CUDA 缓存（可能失败，但不影响继续）
            try:
                torch.cuda.empty_cache()
            except:
                pass
        except Exception as e:
            print(f"Error processing sample {idx}: {type(e).__name__}: {str(e)}")
            try:
                torch.cuda.empty_cache()
            except:
                pass
            predicted_answer = "error"
            output_text = f"Error: {type(e).__name__} - {str(e)[:100]}"

        # 清理视频帧
        if 'video_frames_list' in locals():
            for frames in video_frames_list:
                for frame in frames:
                    del frame
                del frames
            del video_frames_list
        torch.cuda.empty_cache()

        # 保存结果
        result_entry = entry.copy()
        result_entry["predicted"] = {
            "answer": predicted_answer,
            "raw_output": output_text
        }
        results.append(result_entry)

        if skip_for_stats:
            continue

        # 获取正确答案并统计
        chosen_answer = get_chosen_answer(entry.get("chosen", []))

        # 统计模型预测为yes的总数（用于计算查准率）
        if predicted_answer == "yes":
            predicted_yes += 1

        if chosen_answer == "yes":
            total_yes += 1
            if predicted_answer == "yes":
                correct_yes += 1
        elif chosen_answer == "no":
            total_no += 1
            if predicted_answer == "no":
                correct_no += 1

        # 为 AUC / P@K 记录标签与打分（仅在标注为 yes/no 且预测未出错时）
        if chosen_answer in ("yes", "no") and predicted_answer not in (
                "error",
                "error_cuda",
                "error_runtime",
        ):
            label = 1 if chosen_answer == "yes" else 0
            # 置信度：模型越自信地说“yes”，分数越大
            if predicted_answer == "yes":
                score = seq_logprob
            elif predicted_answer == "no":
                score = -seq_logprob
            else:
                # unknown 等情况，给一个中性分数
                score = 0.0
            auc_labels.append(label)
            auc_scores.append(score)

    # 计算统计信息
    stats = {
        "skipped_samples": skipped_samples,
        "cuda_error_samples": cuda_error_samples,
    }
    if total_yes > 0:
        stats["accuracy_yes"] = correct_yes / total_yes * 100  # Recall
        stats["total_yes"] = total_yes
        stats["correct_yes"] = correct_yes
    else:
        stats["accuracy_yes"] = 0.0
        stats["total_yes"] = 0
        stats["correct_yes"] = 0

    # 计算查准率（Precision）：(模型判断为yes且人工判断为yes) / (模型判断为yes的总数)
    if predicted_yes > 0:
        stats["precision_yes"] = correct_yes / predicted_yes * 100
        stats["predicted_yes"] = predicted_yes
    else:
        stats["precision_yes"] = 0.0
        stats["predicted_yes"] = 0

    # 计算误报（False Positives）和 F1
    false_positives = max(predicted_yes - correct_yes, 0)
    stats["false_positives_yes"] = false_positives
    stats["false_negatives_yes"] = max(total_yes - correct_yes, 0)
    precision_val = stats["precision_yes"] / 100 if stats["precision_yes"] > 0 else 0.0
    recall_val = stats["accuracy_yes"] / 100 if stats["accuracy_yes"] > 0 else 0.0
    if precision_val + recall_val > 0:
        stats["f1_yes"] = (
                2 * precision_val * recall_val / (precision_val + recall_val) * 100
        )
    else:
        stats["f1_yes"] = 0.0

    if total_no > 0:
        stats["accuracy_no"] = correct_no / total_no * 100
        stats["total_no"] = total_no
        stats["correct_no"] = correct_no
    else:
        stats["accuracy_no"] = 0.0
        stats["total_no"] = 0
        stats["correct_no"] = 0

    total_samples = total_yes + total_no
    if total_samples > 0:
        total_correct = correct_yes + correct_no
        stats["overall_accuracy"] = total_correct / total_samples * 100
        stats["total_samples"] = total_samples
        stats["total_correct"] = total_correct
    else:
        stats["overall_accuracy"] = 0.0
        stats["total_samples"] = 0
        stats["total_correct"] = 0

    # 真正的 ROC AUC（基于 yes 作为正类）
    if len(auc_labels) > 0 and len(set(auc_labels)) > 1:
        stats["auc_roc_yes"] = compute_roc_auc(auc_labels, auc_scores)
    else:
        stats["auc_roc_yes"] = 0.0

    # 真正的 Top-K Precision：按 score 从高到低取前 K 个
    if top_k is not None and top_k > 0 and len(auc_scores) > 0:
        p_at_k, used_k = compute_p_at_k(auc_labels, auc_scores, top_k)
        stats["p_at_k_yes"] = p_at_k
        stats["p_at_k_k"] = used_k
    else:
        stats["p_at_k_yes"] = 0.0
        stats["p_at_k_k"] = 0

    # 额外记录固定几个 K 的 P@K：50 / 100 / 200 / 500
    for fixed_k in (50, 100, 200, 500):
        key_p = f"p_at_{fixed_k}_yes"
        key_k = f"p_at_{fixed_k}_k"
        if len(auc_scores) > 0:
            p_val, used_k = compute_p_at_k(auc_labels, auc_scores, fixed_k)
            stats[key_p] = p_val
            stats[key_k] = used_k
        else:
            stats[key_p] = 0.0
            stats[key_k] = 0

    return results, stats


def main():
    parser = argparse.ArgumentParser(
        description="推理视频偏好测试集，评估模型性能"
    )
    parser.add_argument(
        "--base_model",
        default="./Qwen3-VL-8B-Instruct",
        help="Qwen3-VL 基座模型路径",
    )
    parser.add_argument(
        "--lora_dir",
        default="/root/autodl-tmp/qwen3vl-lora-video-pref",
        help="LoRA 权重目录（将遍历所有 checkpoint-* 子目录）",
    )
    parser.add_argument(
        "--use_base_model_only",
        action="store_true",
        help="只使用基础模型，不加载 LoRA 权重（用于对比基础模型性能）",
    )
    parser.add_argument(
        "--test_file",
        default="./data/test.json",
        help="测试数据 JSON 路径",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs",
        help="输出目录（所有结果将保存到此目录）",
    )
    parser.add_argument(
        "--frames_per_video",
        type=int,
        default=8,
        help="每个视频提取的帧数（如果遇到 OOM，可以尝试减少到 8 或 6）",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="生成的最大新 token 数",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=25600,
        help="输入长度超过该值的样本将被跳过（0 表示不限制）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批处理大小（推理时建议为1）",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="用于计算 P@K 的 K（Top-K Precision），默认 1000",
    )
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载测试数据
    print(f"加载测试数据: {args.test_file}")
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"测试样本数: {len(test_data)}")

    # 加载模型和处理器
    print(f"加载基座模型: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=False,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    if tokenizer.chat_template is None:
        tmpl_path = Path(args.base_model) / "chat_template.json"
        if tmpl_path.exists():
            tokenizer.chat_template = tmpl_path.read_text()

    # 使用量化配置（与训练时一致）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # 加载基础模型（只加载一次，后续只加载 LoRA 权重）
    print("加载基础模型...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    # 收集所有 checkpoint 路径
    checkpoints = []
    if args.use_base_model_only:
        checkpoints.append(("base_model", None))
    else:
        lora_dir = Path(args.lora_dir)
        if lora_dir.exists():
            # 查找所有 checkpoint-* 目录
            checkpoint_dirs = sorted(
                [d for d in lora_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0
            )
            for checkpoint_dir in checkpoint_dirs:
                checkpoints.append((checkpoint_dir.name, str(checkpoint_dir)))
            print(f"找到 {len(checkpoints)} 个 checkpoint")
        else:
            print(f"Warning: LoRA 目录不存在: {args.lora_dir}")
            checkpoints.append(("base_model", None))

    # 存储所有结果
    all_results_summary = []

    max_len_warning = args.max_len if args.max_len > 0 else None

    # 对每个 checkpoint 运行推理
    for idx, (checkpoint_name, checkpoint_path) in enumerate(checkpoints):
        print(f"\n{'=' * 60}")
        print(f"处理 checkpoint {idx + 1}/{len(checkpoints)}: {checkpoint_name}")
        print(f"{'=' * 60}")

        # 加载模型（如果是基础模型，直接使用；否则加载 LoRA）
        if checkpoint_path is None:
            model = base_model
            model.eval()
        else:
            # 重新加载基础模型（避免 LoRA 权重冲突）
            # 每次都需要重新加载，因为 PeftModel 会修改基础模型
            if idx > 0:  # 第一次已经加载过了，需要先删除
                del base_model
            torch.cuda.empty_cache()
            base_model = AutoModelForVision2Seq.from_pretrained(
                args.base_model,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
            model.eval()

        # 运行推理
        results, stats = run_inference_for_checkpoint(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            test_data=test_data,
            checkpoint_name=checkpoint_name,
            frames_per_video=args.frames_per_video,
            max_new_tokens=args.max_new_tokens,
            max_length=max_len_warning,
            top_k=args.top_k,
        )

        # 保存单个 checkpoint 的结果
        output_file = output_dir / f"{checkpoint_name}_predictions.json"
        print(f"\n保存结果到: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 显示统计信息
        print(f"\n{checkpoint_name} 评估结果:")
        print(f"  总样本数: {stats['total_samples']}")
        print(f"  总正确数: {stats['total_correct']}")
        print(f"  综合正确率: {stats['overall_accuracy']:.2f}%")
        print(f"  'yes' 正确率: {stats['accuracy_yes']:.2f}% ({stats['correct_yes']}/{stats['total_yes']})")
        print(f"  'no' 正确率: {stats['accuracy_no']:.2f}% ({stats['correct_no']}/{stats['total_no']})")
        print(
            f"  'yes' 查准率 (Precision): {stats['precision_yes']:.2f}% ({stats['correct_yes']}/{stats['predicted_yes']})")
        print(f"  'yes' 误报数 (False Positives): {stats['false_positives_yes']}")
        print(f"  'yes' F1: {stats['f1_yes']:.2f}%")
        print(f"  ROC AUC (yes as positive): {stats['auc_roc_yes']:.2f}%")
        print(f"  P@K  (K={stats['p_at_k_k']}): {stats['p_at_k_yes']:.2f}%")
        print(f"  P@50 : {stats['p_at_50_yes']:.2f}% (K={stats['p_at_50_k']})")
        print(f"  P@100: {stats['p_at_100_yes']:.2f}% (K={stats['p_at_100_k']})")
        print(f"  P@200: {stats['p_at_200_yes']:.2f}% (K={stats['p_at_200_k']})")
        print(f"  P@500: {stats['p_at_500_yes']:.2f}% (K={stats['p_at_500_k']})")

        # 添加到汇总
        all_results_summary.append({
            "checkpoint": checkpoint_name,
            "checkpoint_path": checkpoint_path,
            "stats": stats
        })

        # 清理模型（除了最后一个 checkpoint）
        if checkpoint_path is not None and idx < len(checkpoints) - 1:
            del model
            torch.cuda.empty_cache()

    # 保存汇总结果
    summary_file = output_dir / "all_checkpoints_summary.json"
    print(f"\n保存汇总结果到: {summary_file}")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results_summary, f, ensure_ascii=False, indent=2)

    # 显示最终汇总
    print(f"\n{'=' * 60}")
    print("所有 Checkpoint 汇总结果:")
    print(f"{'=' * 60}")
    for result in all_results_summary:
        stats = result["stats"]
        print(f"{result['checkpoint']:30s} | "
              f"Acc: {stats['overall_accuracy']:6.2f}% | "
              f"YesR: {stats['accuracy_yes']:6.2f}% | "
              f"NoR: {stats['accuracy_no']:6.2f}% | "
              f"P(yes): {stats['precision_yes']:6.2f}% | "
              f"FP: {stats['false_positives_yes']:4d} | "
              f"F1: {stats['f1_yes']:6.2f}% | "
              f"AUC: {stats['auc_roc_yes']:6.2f}% | "
              f"P@50: {stats['p_at_50_yes']:6.2f}% | "
              f"P@100: {stats['p_at_100_yes']:6.2f}% | "
              f"P@200: {stats['p_at_200_yes']:6.2f}% | "
              f"P@500: {stats['p_at_500_yes']:6.2f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

