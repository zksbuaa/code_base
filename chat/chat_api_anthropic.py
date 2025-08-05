import sys
import os
from time import sleep
from tqdm import tqdm
import re
import json
import concurrent.futures
import requests
from typing import List, Dict, Optional

personal_base_url = "http://123.129.219.111:3000/v1/chat/completions"
personal_api_key = "sk-JB7EmnGzZwhvKRIjAbMGLb4IIqVRsuucvd7EbtE5t6PoFMh6"

max_try_num = 15
max_thread_num = 8
parse_json = False    # 是否将回答解析成json格式

gpt_model = "gpt-4o-mini"
max_tokens = 4096
temperature = 0

template = """\
{text}
"""

class AnthropicClient:
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com/v1"):
        self.api_key = api_key
        self.base_url = base_url

    class ChatCompletions:
        def __init__(self, client):
            self.client = client

        def create(
            self,
            model: str,
            messages: List[Dict[str, str]],
            temperature: Optional[float] = 1.0,
            max_tokens: Optional[int] = 1024,
            **kwargs,
        ):
            """
            仿 OpenAI 风格的 Anthropic API 调用
            Args:
                model: 模型名称，如 "claude-2.1"
                messages: 消息列表，格式如 [{"role": "user", "content": "Hello"}]
                temperature: 生成温度 (0-1)
                max_tokens: 最大 token 数
                **kwargs: 其他 Anthropic 参数（如 stop_sequences）
            Returns:
                dict: Anthropic API 的响应，包含 "completion" 字段
            """

            # 构造请求数据
            data = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,  # 允许传入其他 Anthropic 参数
            }

            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.client.api_key,
            }
            response = requests.post(
                self.client.base_url,
                headers=headers,
                data=json.dumps(data),
            )
            response.raise_for_status()
            return response.json()

    @property
    def chat(self):
        return self.ChatCompletions(self)
     
     

class ParseJson:
   def __init__(self):
      super().__init__()

   def replace_newlines(self, match):
    # 在匹配的字符串中替换 \n 和 \r
    return match.group(0).replace('\n', '\\n').replace('\r', '\\r')

   def clean_json_str(self, json_str: str) -> str:
      """
      生成的json格式可能不标准，先进行替换处理
      """
      json_str = json_str.replace("None","null")

      # 去除代码块符号```
      #json字符串中None换成null
      json_str = json_str.replace("None","null")

      match = re.search(r'```json(.*?)```', json_str, re.DOTALL)
      if match:
         json_str = match.group(1)
      match = re.search(r'```(.*?)```', json_str, re.DOTALL)
      if match:
         json_str = match.group(1)
      # 在匹配的字符串中替换 \n 和 \r
      json_str = re.sub( r'("(?:\\.|[^"\\])*")', self.replace_newlines, json_str)
      # 移除键值对后面多余的逗号
      json_str = re.sub(r',\s*}', '}', json_str)
      json_str = re.sub(r',\s*]', ']', json_str)
      # 修复缺少的逗号
      json_str = re.sub(r'\"\s+\"', '\",\"', json_str)
      # True、False替换
      json_str = json_str.replace("True","true")
      json_str = json_str.replace("False","false")
      return json_str

   def txt2obj(self, text): 
      try:
         text = self.clean_json_str(text)
         return json.loads(text)
      except Exception as e:
         print(e)
         return None


class ChatCompletion:
   def __init__(self, chunks):
      self.chunks = chunks
      self.template = template
      if parse_json:
         self.parse_json = ParseJson()

   def _get_chat_completion(self, chunk):
      messages = [{"role": "user", "content": template.format(text=chunk)}]
      client = AnthropicClient(api_key=personal_api_key, base_url=personal_base_url)
      chat_completion = client.chat.create(model=gpt_model,
                                          messages=messages,
                                          temperature=temperature,
                                          max_tokens=max_tokens,
                                          frequency_penalty=0,
                                          presence_penalty=0)

      if parse_json:
         return self.parse_json.txt2obj(chat_completion.choices[0].message.content)
      return chat_completion

   
   def get_chat_completion(self, chunk):
      retry = 0
      while retry < max_try_num:
         try:
            return self._get_chat_completion(chunk)
         except Exception as e:
            retry += 1
            sleep(0.1*retry)
            print(e)
      else:
         raise Exception("Max try number reached.")
         return None

   def complete(self):
      with concurrent.futures.ThreadPoolExecutor(max_workers=max_thread_num) as executor:
         future_results = list(tqdm(executor.map(self.get_chat_completion, self.chunks), total=len(self.chunks)))

      return future_results

if __name__ == "__main__":
   chunks = ['''
   Hi, who are you?
   ''']
   chatbot = ChatCompletion(chunks)
   results = chatbot.complete()
   print(results)
   