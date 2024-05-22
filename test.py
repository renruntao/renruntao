import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
try:
  from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
  from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.embeddings import resolve_embed_model
import os
import base64
from streamlit_javascript import st_javascript
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
)
from llama_index.core.indices.query.schema import QueryBundle, QueryType
from llama_index.core.query_engine import RetrieverQueryEngine
import json
import re
import uuid
from typing import Dict, List, Tuple
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms.utils import LLM
from llama_index.core.schema import MetadataMode, TextNode
from tqdm import tqdm
from llama_index.core.schema import NodeWithScore
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set Streamlit page config
st.set_page_config(page_title="NSFC Policy Q&A System", page_icon="", layout="wide", initial_sidebar_state="auto", menu_items=None)
os.environ['OPENAI_API_KEY'] = 'sk-M2QdYLDKUlqq7fow95583aEe565f49C5A0753167436a5396'
os.environ['OPENAI_API_BASE']="https://api.xty.app/v1"

llm = OpenAI(model="gpt-3.5-turbo", system_prompt="用汉语回答我的问题.", temperature=0.7)

# set a global llm
Settings.llm = llm

# App title and image
st.title("Policy Q&A System")

# 初始化聊天引擎
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "您可以咨询任何有关国家自然科学基金条例的相关政策"}
    ]
    
# 用于显示PDF文档的函数
def display_pdf(file_path, page_number=1):
    try:
        with open(file_path, "rb") as file:
            base64_pdf = base64.b64encode(file.read()).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}#page={page_number}"  width="660" height="1000" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except FileNotFoundError:
        st.image('./Scholar.png')

# 加载和索引数据
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        Settings.embed_model = OpenAIEmbedding()
        documents = SimpleDirectoryReader("./NSFc").load_data()
        node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
        nodes = node_parser.get_nodes_from_documents(documents)
        vector_index = VectorStoreIndex(nodes)
        service_context = ServiceContext.from_defaults(llm=None, embed_model=Settings.embed_model)
        vector_index = VectorStoreIndex(nodes, service_context=service_context)
        return vector_index

vector_index = load_data()

class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""
    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
    ) -> None:
        """Init params."""
        self._vector_retriever = vector_retriever
                
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        retrieved_nodes = self._vector_retriever.retrieve(query_bundle)
        return retrieved_nodes
            
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retrieve(query_bundle)
    async def aretrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        return self._aretrieve(str_or_query_bundle)
    
if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k = 10)
        custom_retriever = CustomRetriever(vector_retriever)
        query_engine = RetrieverQueryEngine.from_args(custom_retriever)
        st.session_state.chat_engine = query_engine


    

prompt = st.text_input("请输入您的问题：")
if st.button("提交"):
    col1, col2 = st.columns(spec=[0.43, 0.57], gap="medium")
    if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "您可以咨询任何有关国家自然科学基金条例的相关政策"}]
    st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 生成新的回应
    response = st.session_state.chat_engine.query(prompt)
    first_node = response.source_nodes[0].node
    text = first_node.text.replace('\n', '')
    file_name = first_node.metadata['file_name']
    page_label = first_node.metadata['page_label']
    output = f"""
            **参考答案:** {response.response}\n
            **条规依据:** {text}\n
            **政策来源:** {file_name}\n
            **所在页数:** 第{page_label}页
            """
    st.session_state.current_pdf = f"./NSFc/{file_name}"  # 更新左侧PDF文件路径
    st.session_state.current_page = page_label  # 更新要显示的PDF页面
    with col1:
        with open(st.session_state.current_pdf, "rb") as file:
            base64_pdf = base64.b64encode(file.read()).decode('utf-8')
            pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}#page={page_label}"  width="660" height="1000" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)
    with col2:
        st.write(output)


    



    

