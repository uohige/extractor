import streamlit as st
from slang import LLM, generate_pydantic_model
from unstructured.partition.auto import partition
from loguru import logger
from pydantic import BaseModel, Field
import io
import pandas as pd

st.set_page_config(page_title="Extractor", layout="wide")

user_instruction = st.text_input("抽出指示", placeholder="レシピの中で使われている食材を全て抽出し、それらがどのようなジャンルに分類されるのか整理してください。")

# セッションステートにカラムのリストを初期化
if 'columns' not in st.session_state:
    # 最初に1行分のカラムを追加
    st.session_state.columns = [{'name': '', 'type': '', 'enum': '', 'desc': ''}]

# 各カラムの入力フィールドを表示する関数
def make_column(index):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.session_state.columns[index]['name'] = st.text_input("抽出項目", placeholder="例: 食材のジャンル", key=f"name_{index}")

    with col2:
        st.session_state.columns[index]['type'] = st.selectbox("データ型", ["数値", "文字列"], key=f"type_{index}")

    with col3:
        st.session_state.columns[index]['enum'] = st.text_input("(任意) 選択肢", placeholder="例: 野菜, 肉, 調味料", key=f"enum_{index}")

    with col4:
        st.session_state.columns[index]['desc'] = st.text_input("説明", placeholder="例: レシピの中で使われている食材のジャンル", key=f"desc_{index}")

# 既存のすべてのカラムを表示
for index in range(len(st.session_state.columns)):
    make_column(index)


# カラム追加ボタン
if st.button("抽出項目の追加"):
    st.session_state.columns.append({'name': '', 'type': '', 'enum': '', 'desc': ''})
    st.rerun()  # 画面をリフレッシュ


st.divider()

# ファイルアップロード
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    file = io.BytesIO(uploaded_file.read())

    docs = partition(file=file)
    doc = "\n".join([doc.text for doc in docs])


# 実行ボタン
if st.button("抽出実行", type="primary"):
    with st.spinner("抽出中..."):
        # スキーマを作成
        schema = {}
        for column in st.session_state.columns:
            # enumをリスト形式に変換
            enum_list = column['enum'].split(',') if column['enum'] else []
            schema[column['name']] = {
                "type": "str" if column['type'] == "文字列" else "float",
                "description": column['desc'],
                "enum": enum_list
            }
        # JSON形式で出力
        UserSchema = generate_pydantic_model(schema)

        system_prompt = f"""
        ユーザーが示す情報源から、求められている情報を嘘や矛盾なく正確に抜き出してください。
        情報源から判断できない場合や抽出できない場合はNoneや空欄を返してください。
        与えられたフォーマットに従って、日本語で出力してください。

        また、ユーザーから与えられた以下の指示を参考にしてください。

        # ユーザーからの指示
        - {user_instruction}
        """

        llm = LLM(model="gpt-4o", system_prompt=system_prompt, stream=False, response_format=UserSchema)
        res = llm.chat(user_prompt=doc)

        res = res.choices[0].message.parsed.dict()["fields"]
        df = pd.DataFrame(res)

    st.write(df)
    csv = df.to_csv(index=False).encode("utf-8_sig")

    st.download_button(label="ダウンロード", data=csv, file_name="extracted.csv", mime="text/csv")
