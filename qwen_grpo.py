import re
import torch
import pandas as pd
import torch.distributed as dist
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from modelscope.msdatasets import MsDataset

# 初始化分布式训练
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

if torch.cuda.device_count() > 1:
    dist.init_process_group(backend="nccl")

# 训练系统提示
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# 加载数据集
data = MsDataset.load('testUser/GSM8K_zh', subset_name='default', split='train', cache_dir='/data/felix/GRPO训练')
data = pd.DataFrame(data)[['question_zh', 'answer_only']]

def convert_data(data):
    return [{
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x[0]}
        ],
        'answer': x[1]
    } for x in data.values.tolist()]

dataset = convert_data(data)

# 解析 XML 标签中的 <answer> 部分
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1].split("</answer>")[0]
    return answer.strip()

# 计算回答正确性的奖励
def correctness_reward_func(prompts, completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# 计算回答是否是整数的奖励
def int_reward_func(completions, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

# 检查格式的严格奖励
def strict_format_reward_func(completions, **kwargs):
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

# 检查格式的宽松奖励
def soft_format_reward_func(completions, **kwargs):
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

# 统计 XML 标签奖励
def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
    if text.count("\n</answer>") == 1:
        count += 0.125
    return count

def xmlcount_reward_func(completions, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# 定义模型与训练参数
model_name = "/data/felix/qwen/qwen/Qwen2.5-7B-Instruct"
output_dir = "/data/felix/GRPO训练/outputs/Qwen-7B-GRPO"
run_name = "Qwen-7B-GRPO-gsm8k-zh"

training_args = GRPOConfig(
    report_to="wandb"
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,  # 启用 bf16 减少显存占用
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    num_generations=8,
    max_prompt_length=256,
    max_completion_length=200,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=False,
    vllm_gpu_memory_utilization=0.3,
    vllm_device="cuda:0",
    report_to="none"
)

# 加载模型并分配到多 GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    # device_map="auto"  # 让 transformers 自动分配 GPU
).to("cuda")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 初始化训练器
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func
    ],
    args=training_args,
    train_dataset=dataset
)

# 训练
trainer.train()

# 保存模型
trainer.save_model(output_dir)

# 释放显存
torch.cuda.empty_cache()
