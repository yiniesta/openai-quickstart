import gradio as gr
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

def initialize_autogpt_bot():
    search = SerpAPIWrapper()
    global searchResult
    searchResult = ''
    tools = [
        Tool(
            name="search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        )
    ]
    embeddings_model = OpenAIEmbeddings()
    # OpenAI Embedding 向量维数
    embedding_size = 1536
    # 使用 Faiss 的 IndexFlatL2 索引
    index = faiss.IndexFlatL2(embedding_size)
    # 实例化 Faiss 向量数据库
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    
    global AGENT_BOT
    AGENT_BOT = AutoGPT.from_llm_and_tools(
        ai_name="Eva",
        ai_role="Intelligent Assistant",
        tools=tools,
        llm=ChatOpenAI(temperature=0),
        memory=vectorstore.as_retriever(),  # 实例化 Faiss 的 VectorStoreRetriever
    )
    # 打印 Auto-GPT 内部的 chain 日志
    AGENT_BOT.chain.verbose = True

    return AGENT_BOT

def autogpt_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")

    goals = message.split(";")
    goals.append('the final response should be the search result')
    ans = AGENT_BOT.run(goals)
    return ans
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=autogpt_chat,
        title="AutoGPT智能助手",
        retry_btn=None,
        undo_btn=None,
        chatbot=gr.Chatbot(height=400),
    )

    demo.launch(share=True, server_name="127.0.0.1")

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"
    os.environ["SERPAPI_API_KEY"] = "YOUR_SERPAPI_KEY"
    # 初始化 AutoGPT 机器人
    initialize_autogpt_bot()
    # 启动 Gradio 服务
    launch_gradio()
