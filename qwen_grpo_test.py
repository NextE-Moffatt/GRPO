from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer
import torch
model = AutoModelForCausalLM.from_pretrained("/data/felix/GRPO训练/Qwen/Qwen2.5-0.5B-Instruct").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("/data/felix/GRPO训练/Qwen/Qwen2.5-0.5B-Instruct")

model_r1 = AutoModelForCausalLM.from_pretrained("outputs/Qwen-0.5B-GRPO/checkpoint-1099").to("cuda")
tokenizer_r1 = AutoTokenizer.from_pretrained("outputs/Qwen-0.5B-GRPO/checkpoint-1099")
def generate_with_stream(input_text, model, tokenizer):
    print(f"\n输入: \n{input_text}")
    print("\n输出:")
    
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    streamer = TextStreamer(tokenizer)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=1024,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            streamer=streamer
        )
    
    # 完整结果
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs
input_text = "桑切斯先生发现他五年级学生中有40%的学生最终成绩低于B。如果他有60名五年级学生，有多少学生的最终成绩是B及以上？"
# 使用
print("Qwen:")
qwen_ans = generate_with_stream(input_text, model, tokenizer)
# 使用
print("GRPO:")
r1_ans = generate_with_stream(input_text, model_r1, tokenizer_r1)