import sys
import os
from time import sleep
from tqdm import tqdm
import re
import json
import concurrent.futures
from openai import OpenAI
from typing import Dict, Any

MODEL_CFG: Dict[str, str] = {
   'base_url' : "http://123.129.219.111:3000/v1",
   'api_key' : "sk-J4OU0nswdAQEmN7y7pS9ytPedSvEC8NXCOhuBX5GIz3dXz3c",
   'gpt_model' : "gpt-4o"
}

GENERATE_CFG: Dict[str, Any] = {
   'max_try_num' : 15,
   'max_thread_num' : 8,
   'parse_json' : False,    # 是否将回答解析成json对象
   
   'max_tokens' : 4096,
   'temperature' : 0
}

template = """\
{text}
"""

tools = [{
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
      }]


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
      if GENERATE_CFG['parse_json']:
         self.parse_json = ParseJson()

   def _get_chat_completion(self, chunk):
      messages = [{"role": "user", "content": template.format(text=chunk)}]
      client = OpenAI(api_key=MODEL_CFG['api_key'], base_url=MODEL_CFG['base_url'])
      chat_completion = client.chat.completions.create(model=MODEL_CFG['gpt_model'],
                                                      messages=messages,
                                                      tools=tools,
                                                      temperature=GENERATE_CFG['temperature'],
                                                      max_tokens=GENERATE_CFG['max_tokens'],
                                                      frequency_penalty=0,
                                                      presence_penalty=0)

      if GENERATE_CFG['parse_json']:
         return self.parse_json.txt2obj(chat_completion.choices[0].message.content)
      return chat_completion.choices[0].message.content
      # return chat_completion.choices[0].message

   
   def get_chat_completion(self, chunk):
      retry = 0
      while retry < GENERATE_CFG['max_try_num']:
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
      with concurrent.futures.ThreadPoolExecutor(max_workers=GENERATE_CFG['max_thread_num']) as executor:
         future_results = list(tqdm(executor.map(self.get_chat_completion, self.chunks), total=len(self.chunks)))

      return future_results

if __name__ == "__main__":
   chunks = ['''
   How is the weather in Beijing?
   ''']
   chatbot = ChatCompletion(chunks)
   results = chatbot.complete()
   print(results)
   