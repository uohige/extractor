import os
import json
from pathlib import Path
import pathlib
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, pydantic_function_tool, Stream, NotGiven
from openai.types.chat.chat_completion import ChatCompletion
import tiktoken
import base64
import io
import joblib
import inspect
import magic
import datetime
from typing import Callable, Generator, Any
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo
from loguru import logger

import faiss
import uuid

from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements

NOT_GIVEN = NotGiven()

load_dotenv()


class LLM:
    def __init__(self, 
                 model: str = "gpt-4o-mini", 
                 max_retries: int = 2,
                 timeout: int | None = None,
                 max_completion_tokens: int | None = None,
                 temperature: float = 1.0,
                 stream: bool = True,
                 system_prompt: str | None = None,
                 tools: list | None = None,
                 tool_choice: str = "auto",
                 response_format: BaseModel | None = None
                ):
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.temperature = temperature
        self.stream = stream
        if self.stream==True:
            self.stream_options = {"include_usage": True}
        if self.stream==False:
            self.stream_options = NOT_GIVEN
        self.system_prompt = system_prompt
        if tools is None or tools is NOT_GIVEN:
            self.tools = NOT_GIVEN
            self.tool_choice = NOT_GIVEN
        elif tools is not None or tools is not NOT_GIVEN:
            self.tools = tools.format
            self.tool_choice = tool_choice
        self.response_format = response_format # 対応してclientかclient.betaを使い分ける

        self.client = OpenAI(max_retries=self.max_retries, timeout=self.timeout)

    def chat(self, 
             system_prompt: str | None = None, 
             user_prompt: str | None = None, 
             user_content: list | None = None, 
             messages: list | None = None,
            ):
        """
        Enables you to talk with AI assistant.

        Parameters
        ----------
        system_prompt: str
            prompt characterizing the assistant
        user_prompt: str
            prompt user wants to convey to the assistant
        user_content: list
            
        messages: list

        Returns
        ----------
        response: ChatCompletion | Stream
            received from the assistant
        """
        # LLMに送るプロンプトを入れる
        current_messages = []

        # System Prompt
        if system_prompt is not None:
            self.system_prompt = system_prompt
        
        if self.system_prompt is not None:
            current_messages.extend([{"role": "system", "content": self.system_prompt}])
        if self.system_prompt is None:
            pass

        # Memory
        if messages is not None:
            current_messages.extend(messages)
        if messages is None:
            pass

        # User Prompt
        if user_content is None: 
            user_content = []
            if user_prompt is not None:
                user_content.append({"type": "text", "text": user_prompt})
            if user_prompt is None:
                pass
        if user_content is not None:
            pass

        if bool(user_content)==True:
            user_message = [{"role": "user", "content": user_content}]
            current_messages.extend(user_message)

        # LLMに送信
        if self.response_format is None:
            res = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                stream=self.stream,
                stream_options=self.stream_options,
                messages=current_messages,
                tools=self.tools,
                tool_choice=self.tool_choice
            )
        else:
            res = self.client.beta.chat.completions.parse(
                model=self.model,
                max_completion_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                messages=current_messages,
                tools=self.tools,
                tool_choice=self.tool_choice,
                response_format=self.response_format
            )

        return res
    

class BlankMemory:
    def __init__(self):
        self.messages = []
        self.current_total_tokens = 0

    def add_message(self, role: str | None = None, content: str | list | None = None, message : dict = None):
        pass
    
    def get_messages(self):
        messages = []
        return messages


class Memory:
    def __init__(self):
        self.messages = []
        self.current_total_tokens = 0

    def add_message(self, role: str | None = None, content: str | list | None = None, message : dict = None):
        """
        会話の内容を逐一登録する。

        (例)
        memory.add_message(role="user", content="こんにちは")
        memory.add_message(message={"role":"user", "content": "こんにちは"})

        Parameters
        ----------
        role : str
            発言者のrole
            ["user", "assistant", "tool"]

        content : str | list | None, default=None
            発言内容

        message : dict, default=None
            message形式の発言内容
        """
        # Toolsが選択された場合
        if message is not None:
            self.messages.append(message)
            return None
        
        if bool(content)==False:
            content = None
        
        if type(content)==str:
            message = {"role": role, "content": [{"type": "text", "text": content}]}
            self.messages.append(message)
            
        if type(content)==list:
            message = {"role": role, "content": content}
            self.messages.append(message)

    def get_messages(self):
        """
        過去の記録内容を取得する。

        Returns
        ----------
        messages : list
            過去の記録内容
        """
        return self.messages

    def update_current_total_tokens(self, current_total_tokens):
        """
        現在のtotal_tokensを記録・更新する。

        Parameters
        ----------
        current_total_tokens : int
            現在のtotal_tokens
        """
        self.current_total_tokens = current_total_tokens

    def get_current_total_tokens(self):
        """
        記録されている現在のcurrent_total_tokensを取得する。
        """
        return self.current_total_tokens
    
    def calc_current_tokens_num(self):
        enc = tiktoken.get_encoding("o200k_base")
        
        words = ""
        for m in self.messages:
            words += m["content"]

        current_tokens_num = len(enc.encode(words))
        self.update_current_total_tokens(current_tokens_num)

    def pop(self):
        """
        記録されているmessagesから最後の1つを削除する。
        """
        return self.messages.pop()

    def init(self):
        """
        messagesを初期化する。
        """
        self.messages = []
    
    # def _summarize(self, last_summarized_message_idx: int = None, max_summary_tokens_num: int | None = None):
    #     sum_llm = LLM(
    #         system_prompt=f"これから示す過去の会話を、できるだけ情報量を落とさずに会話の要点を簡潔に分かりやすく要約してください。トークン数が{max_summary_tokens_num}となるように要約してください。", 
    #         max_completion_tokens=max_summary_tokens_num,
    #         stream=False
    #     )
    #     summary_res = sum_llm.chat(user_prompt=str(self.messages[:last_summarized_message_idx]))
    #     summary = str_parse(summary_res)
    #     left_memory = self.messages[last_summarized_message_idx:]
    #     self.messages = [{"role": "assistant", "content": [{"type": "text", "text": f"# これまでの会話の要約\n{summary}\n # 以降はこの要約を前提に会話を継続する"}]}]
    #     self.messages.extend(left_memory)
    
    # def summary_reduce(self, max_tokens_num: int = 10000, max_summary_tokens_num: int = 1000):
    #     enc = tiktoken.get_encoding("o200k_base")
        
    #     words = ""
    #     for i, m in enumerate(self.messages):
    #         tokens_num = len(enc.encode(words))
    #         if tokens_num>=max_tokens_num:
    #             logger.info("要約開始")
    #             self._summarize(last_summarized_message_idx=i, max_summary_tokens_num=max_summary_tokens_num)
    #             #self.calc_current_tokens_num()
    #             break
            
    #         if m["content"][0]["type"]=="text":
    #             words += m["content"][0]["text"]


class Tools:
    def __init__(self):
        self.funcs = {}
        self.format = []

    def add_tool(self, func_model: BaseModel, func: Callable):
        """
        tool_callsで呼び出される関数を登録する。

        Parameters
        ----------
        func_model: BaseModel
            関数の引数を定義したSchema

        func: Callable
            関数
        """
        name = func.__name__
        description = func.__doc__.strip() if func.__doc__ else None
        
        tool_format = pydantic_function_tool(func_model, name=name, description=description)
    
        self.funcs[name] = func
        self.format.append(tool_format)

    def run_tool(self, selected_tool: dict) -> dict:
        """
        tool_callsで選択された関数を実行し、結果をmessage形式で返す

        Parameters
        ----------
        selected_tool : dict
            選択された関数の情報

        Returns
        ----------
        res : dict
            message形式の関数の実行結果
        """
        func_name = selected_tool["function"]["name"]
        args = json.loads(selected_tool["function"]["arguments"])
        tool_call_id = selected_tool["id"]
        
        func = self.funcs[func_name]
        res = {"role": "tool", "content": json.dumps({**args, "result": func(**args)}, ensure_ascii=False), "tool_call_id": tool_call_id}
        return res
    

def get_img_type(img: str | bytes) -> str | None:
    if img.startswith("http"):
        return "url"
    
    try:
        # base64デコードを試みる
        decoded_data = base64.b64decode(img, validate=True)

        # MIMEタイプを取得して、画像かどうかを確認
        mime = magic.Magic(mime=True)
        mime_type = mime.from_buffer(decoded_data)

        # MIMEタイプが画像形式であればTrueを返す
        if mime_type.startswith('image/'):
            return mime_type
        else:
            return None
    except (base64.binascii.Error, ValueError) as e:
        logger.warning(e)
        # base64デコードに失敗した場合はFalseを返す
        return None

def img_to_base64(img_path: str) -> str:
    with open(img_path, "rb") as f:
        file = f.read()
        base64_img = base64.b64encode(file)
        base64_str = base64_img.decode("utf-8")

    return base64_str


def make_content(prompt: str | None = None, imgs: str | list[str] | None = None):
    """
    プロンプトおよび画像を送信するためのcontentを作成する。

    Parameters
    ----------
    prompt : str
        プロンプト

    imgs : str | list[str]
        画像のurl、あるいはbase64形式の画像データ

    Returns
    ----------
    content : dict
        プロンプトおよび画像を含むcontent
    """
    content = []
    if prompt is not None:
        content.append({"type": "text", "text": prompt})
    if prompt is None:
        pass

    if imgs is not None:
        imgs = list(imgs)

        for img in imgs:
            img_type = str(get_img_type(img))
            
            if img_type.startswith("url"):
                content.append({"type": "image_url", "image_url": {"url": f"{img}"}})
            if img_type.startswith("image/"):
                content.append({"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{img}"}})

    return content


def is_stream(response: Any) -> bool:
    """
    与えられたresponseがStream(あるいはGenerator)型かどうかを判定する。

    Parameters
    ----------
    response : Any

    Returns
    ----------
    is_stream : bool
    """
    if type(response)==Stream or inspect.isgenerator(response):
        return True
    elif type(response)!=Stream and not inspect.isgenerator(response):
        return False
    

def _check_tool_calls_from_not_stream(response: ChatCompletion) -> tuple[bool, ChatCompletion]:
    """
    ChatCompletion responseから、tool_callsが呼び出されているかを判断する。

    Parameters
    ----------
    response: ChatCompletion

    Returns
    ----------
    is_tool_calls : bool
        tool_callsが呼び出されているかどうか

    response: ChatCompletion
        もとのresponseをそのまま返す
    """
    new_response = response
    
    if response.choices[0].finish_reason=="stop":
        is_tool_calls = False
        return is_tool_calls, response
    
    if response.choices[0].finish_reason=="tool_calls":
        is_tool_calls = True
        return is_tool_calls, response


def _check_tool_calls_from_stream(response: Stream) -> tuple[bool, Generator]:
    """
    Stream responseから、tool_callsが呼び出されているかを判断する。

    Parameters
    ----------
    response: Stream

    Returns
    ----------
    is_tool_calls : bool
        tool_callsが呼び出されているかどうか

    response: Generator
        もとのresponseと同じイテレーションになるジェネレーターを返す
    """
    chunk = next(response)
    if chunk.choices[0].delta.content=="":
        is_tool_calls = False
    elif chunk.choices[0].delta.content is None:
        is_tool_calls = True

    def generate_new_response(chunk, response):
        yield chunk
        for chunk in response:
            yield chunk

    new_response = generate_new_response(chunk, response)
    
    return is_tool_calls, new_response


def check_tool_calls(response: ChatCompletion | Stream) -> ChatCompletion | Generator:
    """
    responseから、tool_callsが呼び出されているかを判断する。

    Parameters
    ----------
    response: ChatCompletion | Stream

    Returns
    ----------
    is_tool_calls : bool
        tool_callsが呼び出されているかどうか

    response: ChatCompletion | Generator
        もとのresponseと同じイテレーションを持つGeneratorを返す
    """
    if is_stream(response):
        is_tool_calls, response = _check_tool_calls_from_stream(response)

    elif not is_stream(response):
        is_tool_calls, response = _check_tool_calls_from_not_stream(response)

    return is_tool_calls, response


def _get_selected_tools_from_not_stream_response(response: ChatCompletion) -> dict:
    """
    ChatCompletion responseから、選択されたtool_callsオブジェクトを取り出す。

    Parameters
    ----------
    response: ChatCompletion

    Returns
    ----------
    selected_tools : dict
        選択されたtool_calls
    """
    selected_tools = response.choices[0].message.to_dict()
    return selected_tools


def _get_selected_tools_from_stream_response(response: Stream) -> dict:
    """
    Stream responseから、選択されたtool_callsオブジェクトを取り出す。

    Parameters
    ----------
    response: Stream

    Returns
    ----------
    selected_tools : dict
        選択されたtool_calls
    """
    tool_calls = []
    for chunk in response:
        if not chunk.choices:
            current_total_tokens = chunk.usage.total_tokens
            continue
        
        delta = chunk.choices[0].delta
        
        if delta.role=="assistant" and delta.tool_calls is None:
            continue
    
        if delta.tool_calls is not None:
            tool_call = delta.tool_calls[0]
    
            if tool_call.id is not None:
                tool_call_id = tool_call.id
                tool_call_idx = tool_call.index
                tool_call_name = tool_call.function.name
                tool_calls.append({"id": tool_call_id, "type": "function", "function": {"name": tool_call_name, "arguments": ""}})
            tool_calls[tool_call_idx]["function"]["arguments"] += tool_call.function.arguments

    selected_tools = {"role": "assistant", "tool_calls": tool_calls}

    return selected_tools


def get_selected_tools(response: ChatCompletion | Stream) -> dict:
    """
    ChatCompletion responseおよびStream responseから、選択されたtool_callsオブジェクトを取り出す。

    Parameters
    ----------
    response: ChatCompletion | Stream

    Returns
    ----------
    selected_tools : dict
        選択されたtool_calls
    """
    if is_stream(response):
        selected_tools = _get_selected_tools_from_stream_response(response)

    elif not is_stream(response):
        selected_tools = _get_selected_tools_from_not_stream_response(response)

    return selected_tools


class Parser:
    def __init__(self):
        self.current_content = ""
        self.current_total_tokens = None
            
    def _parse_stream_response(self, response: Stream) -> Generator[str, None, None]:
        """
        Stream responseから、Assistantのメッセージを取り出す。
        取り出したメッセージはcurrent_contentに保存および戻り値として返す。
        また、現在までのtotal_tokensを取得し、current_total_tokensに保存する。
    
        Parameters
        ----------
        response: Stream
    
        Returns
        ----------
        parsed_response : Generator[str, None, None]
            AssistantのメッセージのGenerator
        """
        self.current_content = ""
        for chunk in response:
            if (bool(chunk.choices)==True) and (chunk.usage is None): # 文章を生成中の場合
                content = chunk.choices[0].delta.content
                if content is None: # contentがNoneの場合はスキップ
                    continue
                self.current_content += content
                yield(content)
            
            if (bool(chunk.choices)==False) and (chunk.usage is not None): # current_total_tokensを出力した場合
                total_tokens = chunk.usage.total_tokens
                self.current_total_tokens = total_tokens

    def _parse_not_stream_response(self, response: ChatCompletion) -> str:
        """
        ChatCompletion responseから、Assistantのメッセージを取り出す。
        取り出したメッセージはcurrent_contentに保存および戻り値として返す。
        また、現在までのtotal_tokensを取得し、current_total_tokensに保存する。
    
        Parameters
        ----------
        response: ChatCompletion
    
        Returns
        ----------
        parsed_response : str
            Assistantのメッセージ
        """
        content = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        self.current_content = content
        self.current_total_tokens = total_tokens
        return content

    def parse_response(self, response) -> str | Generator:
        """
        ChatCompletion responseおよびStream responseから、Assistantのメッセージを取り出す。
        取り出したメッセージはcurrent_contentに保存および戻り値として返す。
        また、現在までのtotal_tokensを取得し、current_total_tokensに保存する。
    
        Parameters
        ----------
        response: ChatCompletion | Stream
    
        Returns
        ----------
        parsed_response : str | Generator
            Assistantのメッセージ
        """
        if is_stream(response):
            response = self._parse_stream_response(response)
            return response
        
        elif not is_stream(response):
            response = self._parse_not_stream_response(response)
            return response
        

class UserAgent:
    def __init__(self, memory):
        self.memory = memory

    def chat(self, prompt=None, imgs=None):
        content = make_content(prompt, imgs)
        self.memory.add_message(role="user", content=content)

class AiAgent:
    def __init__(self, llm, memory=NotGiven, tools=NotGiven):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.parser = Parser()

    def chat(self):
        res = self.llm.chat(messages=self.memory.get_messages())
        is_tool_calls, res = check_tool_calls(res)

        while is_tool_calls:
            logger.info("tool_calls")
            selected_tools = get_selected_tools(res)
            self.memory.add_message(message=selected_tools)
            for selected_tool in selected_tools["tool_calls"]:
                func_res = self.tools.run_tool(selected_tool)
                self.memory.add_message(message=func_res)
        
            res = self.llm.chat(messages=self.memory.get_messages())
            is_tool_calls, res = check_tool_calls(res)

        parsed_res = self.parser.parse_response(res)

        if is_stream(parsed_res):
            for content in parsed_res:
                print(content, end="", flush=True)
        elif not is_stream(parsed_res):
            print(parsed_res)

        self.memory.add_message(role="assistant", content=self.parser.current_content)
        self.memory.update_current_total_tokens(self.parser.current_total_tokens)



class Task(BaseModel):
    task_id: int = Field(..., description="タスクの順番を1から記述する")
    description: str = Field(..., description="タスクの簡潔かつ具体的な指示(ただし関数名は記述しないこと)")

class AllTasks(BaseModel):
    all_tasks: list[Task] = Field(..., description="各タスクを順番にまとめたリスト")

class Planner:
    def __init__(self, memory=NOT_GIVEN, tools=NOT_GIVEN):
        self.system_prompt = """
        与えられたユーザーの要求に完璧に応えるため、LLMが1度に実行できる小さなタスクに分割して完璧な行動計画を立て、指示してください。
        以下の条件に従ってください。
        
        # 条件  
        - 一度に実行可能な簡潔で具体的なタスクを指示する。
        - 可能な限りタスクの数は少なくする。
        - function callingは、可能な限り並列に一度に実行する。
        - 必ず最後までタスクを出力する。
        - 与えられたフォーマットに従って出力する。

        # 例
        - ユーザーが「社員番号123456の社員の現在のスケジュールを調べて。もし空いているなら同様に社員番号777777についても調べて。」と要求した場合の出力例
        {
          "all_tasks": [
            {
              "task_id": 1,
              "description": "社員番号123456と社員番号777777の今週のスケジュールを取得してください。同時に、現在の日時を取得してください。"
            },
            {
              "task_id": 2,
              "description": "取得した社員番号123456と社員番号777777の今週のスケジュールから、現在の日時と一致するスケジュールを抜き出して教えてください。"
            },
          ]
        }
        
        """

        self.system_prompt_for_re_thinking = """
        以前の行動計画には指摘されたような課題がありました。
        指摘内容を参考にして、追加の行動計画を立案してください。
        また、以下の条件に従ってください。
        
        # 条件  
        - 一度に実行可能な簡潔で具体的なタスクを指示する。
        - 可能な限りタスクの数は少なくする。
        - function callingやAPIの呼び出しは、可能な限り並列に一度に実行する。
        - 必ず最後までタスクを出力する。
        - 与えられたフォーマットに従って出力する。

        # 例
        - ユーザーが「社員番号123456の社員の現在のスケジュールを調べて。もし空いているなら同様に社員番号777777についても調べて。」と要求した場合の出力例
        {
          "all_tasks": [
            {
              "task_id": 1,
              "description": "社員番号123456と社員番号777777の今週のスケジュールを取得してください。同時に、現在の日時を取得してください。"
            },
            {
              "task_id": 2,
              "description": "取得した社員番号123456と社員番号777777の今週のスケジュールから、現在の日時と一致するスケジュールを抜き出して教えてください。"
            },
          ]
        }
        
        """
        if memory is NOT_GIVEN:
            self.memory = BlankMemory()
        elif memory is not NOT_GIVEN:
            self.memory = memory
        self.tools = tools
        self.parser = Parser()
        self.llm = LLM(model="gpt-4o",  
               tools=self.tools, 
               tool_choice="none", 
               max_completion_tokens=1024, 
               stream=False,
               response_format=AllTasks,
              )
    
    def think(self, user_prompt: str, re_thinking: bool = False):
        if re_thinking is False:
            system_prompt = self.system_prompt
            self.memory.add_message(role="user", content=user_prompt)
        elif re_thinking is True:
            system_prompt = self.system_prompt_for_re_thinking

        res = self.llm.chat(system_prompt=system_prompt, messages=self.memory.get_messages())
        plan = res.choices[0].message.parsed.dict()
        self.memory.add_message(role="assistant", content=str(plan))
        return plan


class Worker:
    def __init__(self, memory=NOT_GIVEN, tools=NOT_GIVEN):
        if memory is NOT_GIVEN:
            self.memory = BlankMemory()
        elif memory is not NOT_GIVEN:
            self.memory = memory        
        self.tools = tools
        self.llm = LLM(model="gpt-4o-mini", system_prompt="ユーザーに与えられた指示を的確に実行してください。ただし、適切なfunctionやAPIが提供されていないなどの理由で実行不可能な場合は、その旨を説明してください。", tools=self.tools, stream=False)
        self.parser = Parser()
        
    def conduct(self, instruction):
        logger.info(instruction)
        self.memory.add_message(role="user", content=instruction)
        res = self.llm.chat(messages=self.memory.get_messages())
        is_tool_calls, res = check_tool_calls(res)

        if is_tool_calls:
            selected_tools = get_selected_tools(res)
            self.memory.add_message(message=selected_tools)
            for selected_tool in selected_tools["tool_calls"]:
                tool_res = self.tools.run_tool(selected_tool)
                self.memory.add_message(message=tool_res)
                logger.info(tool_res)
        else:
            parsed_res = self.parser.parse_response(res)
            self.memory.add_message(role="assistant", content=parsed_res)
            logger.info(parsed_res)



class CheckedAnswerSchema(BaseModel):
    valid: bool = Field(..., description="LLMの回答が適切である場合はTrue, 不適切な場合はFalseを返す", enum=[True, False])
    reason: str = Field(..., description="LLMの回答が不適切な場合に改善点を記述する")
    
class Checker:
    def __init__(self, memory=NOT_GIVEN, tools=NOT_GIVEN):
        if memory is NOT_GIVEN:
            self.memory = BlankMemory()
        elif memory is not NOT_GIVEN:
            self.memory = memory
        self.tools = tools
        self.llm = LLM(model="gpt-4o", tools=self.tools, tool_choice="none", stream=False, response_format=CheckedAnswerSchema)

    def check(self, user_prompt):
        system_prompt = f"""
        ユーザーの要求に対して、これまでの会話履歴で得られた情報を参考にユーザーに返答すべきかどうかを判断してください。
        現時点で回答するべきと判断した場合は、validにTrueを返し、adciceに"none"と記してください。
        まだ回答するべきではないと判断した場合は、validにFalseを返し、adviceに改善点を記してください。

        # ユーザーの要求
        - {str(user_prompt)}
        """
        res = self.llm.chat(system_prompt=system_prompt, messages=self.memory.get_messages())
        parsed_res = res.choices[0].message.parsed.dict()
        self.memory.add_message(role="assistant", content=str(parsed_res))
        return parsed_res


class ReactAgent:
    def __init__(self, memory=NOT_GIVEN, tools=NOT_GIVEN, stream=False):
        self.memory = memory
        self.shared_memory = Memory()
        self.tools = tools
        self.llm = LLM(model="gpt-4o", system_prompt="あなたはハイテンションな女子高生です。", stream=stream)
        self.planner = Planner(memory=self.shared_memory, tools=self.tools)
        self.worker = Worker(memory=self.shared_memory, tools=self.tools)
        self.checker = Checker(memory=self.shared_memory, tools=self.tools)
        self.summarizer = LLM(model="gpt-4o-mini", stream=False)
        self.parser = Parser()

    def _react(self, user_prompt, re_thinking: bool = False):
        plan = self.planner.think(user_prompt=user_prompt, re_thinking=re_thinking)
        print("plan:", plan)

        for task in plan["all_tasks"]:
            self.worker.conduct(instruction=task["description"])
        
        checked_res = self.checker.check(user_prompt=user_prompt)
        print(checked_res)
    
        return checked_res

    
    def chat(self, user_prompt):
        self.memory.add_message(role="user", content=user_prompt)

        re_thinking = False
        for i in range(5):
            logger.warning(f"user_prompt:{user_prompt}, re_thinking:{re_thinking}")
            react_res = self._react(user_prompt=user_prompt, re_thinking=re_thinking)
            if react_res["valid"]==True:
                break
            elif react_res["valid"]==False:
                re_thinking=True
                continue
                
        # summarized_res = self.summarizer.chat(
        #     system_prompt=f"これまでに得られた情報を元に、次の問い合わせに的確かつ簡潔に回答してください。また、回答不可能な内容に関しては理由を説明してください。\nQ:{user_prompt}", 
        #     messages=self.shared_memory.get_messages()
        # )
        # parsed_summarized_res = self.parser.parse_response(summarized_res)
        
        extended_user_prompt = f"""
        次のユーザーからの問い合わせに対して適切に回答してください。
        参考情報にそのまま答えが記されている場合があります。
        必要があれば参考情報を活用して回答してください。

        # 問い合わせ内容  
        - {user_prompt}

        # 参考情報
        - {str(self.shared_memory.get_messages())}
        """
        self.shared_memory.init()
        
        logger.info(extended_user_prompt)
        res = self.llm.chat(user_prompt=extended_user_prompt, messages=self.memory.get_messages())
        parsed_res = self.parser.parse_response(res)
        if is_stream(parsed_res):
            for chunk in parsed_res:
                print(chunk, end="", flush=True)
        else:
            print(parsed_res)

        self.memory.add_message(role="assistant", content=self.parser.current_content)


def generate_pydantic_model(schema: dict):
    """
    schemaを元にpydanticのデータモデルを生成する。

    Parameters
    ----------
    schema : dict
        pydanticのデータモデルを記述したdict
        
        (例) 
        schema = {
            "Name": {"type": "str", "description": "材料の名前", "enum": []},
            "Hs": {"type": "float", "description": "その材料の硬度", "enum": []},
            "Temp": {"type": "int", "description": "その材料の加硫温度", "enum": ["100", "120"]}
        }

    Returns
    ----------
    UserSchema: class
        pydanticのデータモデル
    """
    class_str = "class UserSubSchema(BaseModel):\n"
    for key, value in schema.items():
        if value["enum"]:
            class_str += f'    {key}: {value["type"]} = Field(..., description="{value["description"]}", enum={str(value["enum"])})\n'
        else:
            class_str += f'    {key}: {value["type"]} = Field(..., description="{value["description"]}")\n'

    namespace = {}
    exec(class_str, globals(), namespace)
    UserSubSchema = namespace["UserSubSchema"]

    class UserSchema(BaseModel):
        fields: list[UserSubSchema] = Field(...)

    return UserSchema


class Vectorizer:
    def __init__(self, 
                 model: str = "text-embedding-3-small", 
                 max_retries: int = 2,
                 timeout: int | None = None,
                 dimensions: int = 1536
                ):
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.dimensions = dimensions
        self.client = OpenAI(timeout=self.timeout, max_retries=self.max_retries)
        
    def embed(self, input: str | list[str] | None = NOT_GIVEN) -> list[list[float]]:
        res = self.client.embeddings.create(input=input, model=self.model, dimensions=self.dimensions)
        embeddings = [data.embedding for data in res.data]
        return embeddings


class VectorDB:
    def __init__(self, 
                 vectordb: str | Any | None = None, 
                 documentdb: str | Any | None = None, 
                 vectorizer: Vectorizer | None = None
                ):
        self.vectordb = vectordb
        self.documentdb = {}
        self.ids_to_keys = {}
        self.current_maps_size = 0
        self.vectorizer = vectorizer

    def init_vectordb(self, dimensions: int = 1536):
        self._index = faiss.IndexFlatL2(dimensions)
        self.vectordb = faiss.IndexIDMap2(self._index)

    def _init_maps(self):
        self.ids_to_keys = {}

    def _add_keys_and_ids(self, keys: str | int | list[str | int], ids: int | list[int]):
        for key, id_ in zip(keys, ids):
            self.ids_to_keys[id_] = key
        self.current_maps_size = len(self.ids_to_keys)

    def add_documents(self, 
                      contents: str | int | float | list[str | int | float], 
                      keys: str | int | list[str | int] | None = None,
                      metadata: dict[str, str | int | float] | None = None
                     ):
        if not isinstance(contents, list):
            contents = [contents]

        if keys is None:
            keys = [uuid.uuid4().hex for i in range(len(contents))]
        elif not isinstance(keys, list):
            keys = [keys]

        assert len(contents)==len(keys)
             
        embeddings = self.vectorizer.embed(contents)
        ids = list(range(self.current_maps_size, self.current_maps_size + len(embeddings)))

        for key, content, embedding in zip(keys, contents, embeddings):
            self.documentdb[key] = {"content": content, "embedding": embedding, "metadata": metadata}

        self.vectordb.add_with_ids(np.atleast_2d(embeddings), ids)
        self._add_keys_and_ids(keys, ids)

    def _get_keys_and_embeddings_from_documentdb(self):
        keys = []
        embeddings = []
        for key, doc in self.documentdb.items():
            keys.append(key)
            embeddings.append(doc["embedding"])
        return keys, embeddings

    def build_vectordb_from_documentdb(self):
        self.init_vectordb()
        self._init_maps()
        
        keys, embeddings = self._get_keys_and_embeddings_from_documentdb()
        ids = list(range(len(keys)))

        self._add_keys_and_ids(keys, ids)
        self.vectordb.add_with_ids(embeddings, ids)
    
    def _get_similar_embedding_ids(self, embeddings: list[float] | list[list[float]], k: int) -> list:
        embeddings = np.atleast_2d(embeddings)
        dists, ids = self.vectordb.search(embeddings, k)
        ids = ids.flatten().tolist()
        return ids

    def get_documents_by_ids(self, ids: int | list[int]):
        if not isinstance(ids, list):
            ids = [ids]

        keys = [self.ids_to_keys[id_] for id_ in ids]
        documents = [self.documentdb[key] for key in keys]

        return documents

    def retrieve_similar_documents(self, content: str, k: int = 3) -> list:
        embedding = self.vectorizer.embed(content)
        ids = self._get_similar_embedding_ids(embedding, k)
        documents = self.get_documents_by_ids(ids)
        
        return documents

    def save(self, dir_path: str | Path = "./"):
        dir_path = pathlib.Path(dir_path)
        vectordb_path = str(dir_path / "vectordb")
        documentdb_path = str(dir_path / "documentdb")
        map_path = str(dir_path / "map")

        if not dir_path.exists():
            dir_path.mkdir(parents=True)
        
        faiss.write_index(self.vectordb, vectordb_path)
        joblib.dump(self.documentdb, documentdb_path)
        joblib.dump(self.keys_to_ids, documentdb_path)

    def load(self, dir_path: str | Path = "./"):
        dir_path = pathlib.Path(dir_path)
        vectordb_path = str(dir_path / "vectordb")
        documentdb_path = str(dir_path / "documentdb")
        
        self.vectordb = faiss.read_index(vectordb_path)
        self.documentdb = joblib.load(documentdb_path)
        assert self.vectordb.ntotal==len(self.documentdb)