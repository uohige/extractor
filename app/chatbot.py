from slang import LLM, Memory, Parser, Tools, UserAgent, AiAgent, check_tool_calls, is_stream, NotGiven, get_selected_tools
import streamlit as st
from loguru import logger
from zoneinfo import ZoneInfo
import datetime
from pydantic import BaseModel, Field
import time

# キャッシュの定義
if "memory" in st.session_state:
    memory = st.session_state.memory
    logger.info(memory.get_messages())
else:
    memory = Memory()
    st.session_state["memory"] = memory

# 関数の定義
def get_current_time(timezone: str = "Asia/Tokyo") -> str:
    """
    指定したタイムゾーンの現在時刻を取得する。

    Parameters
    ----------
    time_zone: str, default="Asia/Tokyo"
        タイムゾーンの指定

    Returns
    ----------
    current_time: str
        %Y/%m/%d %H:%M:%S形式の現在時刻
    """
    timezone = ZoneInfo(timezone)
    current_time = "tool_calls実行時の日時:" + datetime.datetime.now(timezone).strftime("%Y/%m/%d %H:%M:%S")

    return current_time


class GetCurrentTimeSchema(BaseModel):
    timezone: str = Field(..., description="タイムゾーン 例:'Asia/Tokyo'")


tools = Tools()
tools.add_tool(GetCurrentTimeSchema, get_current_time)


# エージェントの定義
class AiAgent:
    def __init__(self, llm, parser, memory=NotGiven, tools=NotGiven):
        self.llm = llm
        self.parser = parser
        self.memory = memory
        self.tools = tools

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
        
        return parsed_res


llm = LLM(model="gpt-4o", system_prompt="あなたはハイテンションな女子高生です！", tools=tools)
parser = Parser()
ai = AiAgent(llm=llm, parser=parser, memory=memory, tools=tools)
user = UserAgent(memory=memory)


# Side Menu
with st.sidebar:
    st.markdown("## 会話履歴をリセット")
    if st.button("リセット"):
        st.session_state.clear()

# Main UI
if user_prompt := st.chat_input("Say something"):
    for message in memory.get_messages():
        if message is None:
            break

        if message["role"]=="user":
            st.chat_message("user").write(message["content"][0]["text"])
        if message["role"]=="assistant":
            if "tool_calls" in message:
                continue
            st.chat_message("assistant").write(message["content"][0]["text"])
    
    st.chat_message("user").write(user_prompt)
    user.chat(prompt=user_prompt)
    st.session_state["memory"] = memory

    parsed_res = ai.chat()
    st.chat_message("assistant").write_stream(parsed_res)
    memory.add_message(role="assistant", content=parser.current_content)
    memory.update_current_total_tokens(current_total_tokens=parser.current_total_tokens)
    st.session_state["memory"] = memory
