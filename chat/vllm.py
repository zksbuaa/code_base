from vllm import LLM, SamplingParams

# 定义采样参数
sampling_params = SamplingParams(temperature=0, max_tokens=200)

# 初始化模型
llm = LLM(model="/mnt/public/model/zks/Qwen2.5-0.5B")

# 定义输入
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is",
]

# 生成文本
outputs = llm.generate(prompts, sampling_params)

# 打印输出
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")