from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/mnt/public/model/zks/Qwen2.5-7B-Instruct"
device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

system_message = '''
You are an expert in using functions (i.e., tools) to solve
users’ tasks. The functions available for you to use are
detailed below:


<tool>
[
    {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定位置的当前天气",
        "parameters": {
        "type": "object",
        "properties": {
            "location": {
            "type": "string",
            "description": "城市名称,如: 北京"
            },
            "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": ["location"]
        }
    }
    },
    {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "计算两个数字的和",
        "parameters": {
        "type": "object",
        "properties": {
            "number1": {
            "type": "float",
            "description": "第一个数字"
            },
            "number2": {
            "type": "float",
            "description": "第二个数字"
            }
        },
        "required": ["number1", "number2"]
        }
    }
    }
]
</tool>


In your response, you need first provide your observation and
thought on the user’s task, the current situation, and what
you plan to do next. After your thinking, you can do
following two things:

**Function Call**: For fountion calling, you need to provide the
function name and its arguments. The function name must be
same as its name in above function list, and the arguments
must obey the format required by the function. Enclose the
function call within the tag "<call></call>". If possible,
you can call multiple functions in parallel, be sure the
functions called parallelly are independent of each other.
**Final Answer**: When you believe the task is complete, you may
use ’final_answer’ to provide a detailed summary of the
results to give to the user, enclose the final answer within
the tag "<final></final>".


'''

prompt = \
'''
Let $\\mathbf{a}$ and $\\mathbf{b}$ be vectors such that\n\\[\\mathbf{v} = \\operatorname{proj}_{\\mathbf{a}} \\mathbf{v} + \\operatorname{proj}_{\\mathbf{b}} \\mathbf{v}\\]for all vectors $\\mathbf{v}.$  Enter all possible values of $\\mathbf{a} \\cdot \\mathbf{b},$ separated by commas.
翻译成中文并解答
'''


messages = [
    # {"role": "system", "content": system_message},
    {"role": "user", "content": prompt},
    #{"role": "assistant", "content": '我将查询北京今天的天气情况。使用 `get_weather` 函数来获取北京的当前天气信息。<call>get_weather(location="北京", unit="celsius")</call>'},
    #{"role": "tool", "content":'[ {"name": "get_weather", "arguments": {"location": "北京", "unit": "celsium"}, "result": "14" } ]'}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
print(text)
print(model_inputs)
print(tokenizer.convert_ids_to_tokens(model_inputs.input_ids[0]))

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048,
    do_sample=False
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
print(response)
print(tokenizer.convert_ids_to_tokens(generated_ids[0]))