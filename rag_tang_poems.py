# -*- coding: utf-8 -*-
"""
Retrieval Augmented Generation For Tang Poems
Query -> Translation&Construction -> Retrieval -> Filter&Rerank -> Generation
"""
import traceback
from typing import List, Iterator, AsyncIterator

import pymilvus.exceptions
from langchain_core.documents import Document
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

TRANSLATION_AND_CONSTRUCTION_PROMPT = """你现在是一个AI Milvus查询条件生成助手，你基于输入的问题输出合理的milvus查询条件。

**背景与处理规则**
现在有一个存储了诗歌数据的Milvus向量数据库，你的任务是将原始关于诗歌```问题```转换为用来查询Milvus数据库的query和expr条件。
参考下面的要求和数据模型完成query和expr的生成任务。
改写```问题```成查询query，使其以更好角度地从包含标题、正文、赏析的诗词向量数据库中搜索到有关问题的上下文信息。比如将"静夜思的作者"改写成"静夜思"
如果不需要进行标量过滤，那么将expr置空。
query字段只需包含改写后用于生成词嵌入(Embedding)的文本，不要包含其它查询参数。
expr字段要符合Milvus标量过滤(Scalar Filter)的语法规则。

**参考数据模型**
| 字段 | 类型 |含义 |
|---  | --- | --- |
|author | varchar | 作者 |
|vector | vector  | 向量，此处的向量可能是下面的一种：诗歌全文、标题、诗的白话文翻译、对诗的赏析 |
|text   | varchar | vector对应的文本，此文的文本可能是下面的一种：诗歌全文、 诗的白话文翻译、诗的赏析文字，不允许对text使用==的标量过滤条件 |

author、vector、text都有值，不会为空。

**输出格式与要求**
- 你必须以JSON格式给出改写后的查询query 和 标量expr 条件，格式如下
```json
{{
  "query": "作者是李白的诗",
  "expr": "author == '李白' "
}}
```

请严格根据上面给出的背景、处理规则、Milvus数据模型以及输出要求来完成下面````问题```的改写：
{question}

!!不要输出任何推理步骤、过程、注释!!
"""

GENERATION_WITH_CONTEXT_PROMPT = """你是一个AI诗词专家，你擅长回答关于诗词的问题。

请仅根据下面的知识信息来回答问题：
{context}

问题:
{question}

注意：
如果你无法回答问题，请回答"对不起 我现在无法回答这个问题"。
你只能回答与诗、诗词、歌赋、诗人相关的问题。
你必须拒绝回答涉及暴力、色情问题。
"""

GENERATION_WITHOUT_CONTEXT_PROMPT = """你是一个AI诗词专家，你擅长回答关于诗词的问题。

请回答下面的问题：
{question}

注意：
如果你无法回答问题，请回答"对不起 我现在无法回答这个问题"。
你只能回答与诗、诗词、歌赋、诗人相关的问题。
你必须拒绝回答涉及暴力、色情问题。
"""


def reciprocal_rank_fusion(results: List[List[Document]], k=60):
    fused_scores = {}
    docs_dict = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_id = doc.metadata['pid']
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                docs_dict[doc_id] = doc
            fused_scores[doc_id] += 1 / (rank + k)
    # sort reverse by fused scores and return the document list
    reranked_results = [
        docs_dict[doc_id]
        for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


def remove_duplicated_documents(results: List[Document]):
    doc_id_set = set()
    docs = []
    for doc in results:
        doc_id = doc.metadata['pid']
        if doc_id not in doc_id_set:
            doc_id_set.add(doc_id)
            docs.append(doc)
    return docs


def build_document_content_from_poem_details(poem_detail: dict) -> str:
    content = f"诗词\n诗名：{poem_detail.get('title', '')}\n"
    content += f"作者：{poem_detail.get('author', '')}\n"
    content += f"诗词：\n{poem_detail.get('content', '')}\n"
    content += f"翻译：\n{poem_detail.get('translation', '')}\n"
    content += f"赏析：{poem_detail.get('appreciation', '')}\n"
    return content


class RagTangPoems(BaseModel):

    class Config:
        arbitrary_types_allowed=True

    llm_for_answer: LLM = Field(description="language model for the answer")
    llm: LLM = Field(description="language model for the query rewrite")
    vector_store: VectorStore = Field(description="vector store")
    search_args: dict = Field(default=None, description="search args for vector store")
    query_rewrite_prompt_template: PromptTemplate = Field(default=None, description="prompt template for query rewrite")
    answer_prompt_template: PromptTemplate = Field(default=None, description="prompt template for answer generation")
    answer_without_context_prompt_template: PromptTemplate = Field(default=None, description="prompt template for answer generation without context")
    timeout: int = Field(default=30, description="timeout for each step")

    poems_store: dict = Field(description="poems store")
    max_token_chars: int = Field(default=1024 * 4, description="max token chars for generate answer")

    @root_validator()
    def _build_default_fields(cls, values: dict) -> dict:
        if not values['query_rewrite_prompt_template']:
            values['query_rewrite_prompt_template'] = PromptTemplate.from_template(TRANSLATION_AND_CONSTRUCTION_PROMPT)
        if not values['answer_prompt_template']:
            values['answer_prompt_template'] = PromptTemplate.from_template(GENERATION_WITH_CONTEXT_PROMPT)
        if not values['answer_without_context_prompt_template']:
            values['answer_without_context_prompt_template'] = PromptTemplate.from_template(GENERATION_WITHOUT_CONTEXT_PROMPT)
        if not values['search_args']:
            values['search_args'] = {
                'top_k': 20,
                'expr': '',
                'param': {
                    'ef': 300
                }
            }
        return values

    def query_rewrite(self, question: str) -> dict:
        """
        rewrite query and construct expr for milvus query
        :param question: question
        :return: {"query": "query", "expr": "expr", "error": "if exists"}
        """
        try:
            chain = self.query_rewrite_prompt_template | self.llm | JsonOutputParser()
            query_dict = chain.invoke({"question": question})
            # remove expr when scalar expr contains text == "xxx"
            if 'text ==' in query_dict.get('expr', ''):
                query_dict['expr'] = ''
            return query_dict
        except:
            # catch all exceptions
            return {
                "query": question,
                "expr": "",
                "error": traceback.format_exc()
            }

    async def aquery_rewrite(self, question: str) -> dict:
        """
        rewrite query and construct expr for milvus query
        :param question: question
        :return: {"query": "query", "expr": "expr", "error": "if exists"}
        """
        try:
            chain = self.query_rewrite_prompt_template | self.llm | JsonOutputParser()
            query_dict = await chain.invoke({"question": question})
            # remove expr when scalar expr contains text == "xxx"
            if 'text ==' in query_dict.get('expr', ''):
                query_dict['expr'] = ''
            return query_dict
        except:
            # catch all exceptions
            return {
                "query": question,
                "expr": "",
                "error": traceback.format_exc()
            }

    def retrieve(self, query: str, expr: str, top_k: int = 20, ef: int = 300) -> List[Document]:
        """
        retrieval from vector store
        :param query: query
        :param expr: expr
        :param top_k: top k
        :param ef: ef
        :return: list of dict
        """
        if not query:
            raise ValueError("query is empty")
        try:
            search_args = self.search_args | {
                "expr": expr,
                "k": top_k,
                "param": {
                    "ef": ef
                }
            }
            retriever: VectorStoreRetriever = self.vector_store.as_retriever(search_kwargs=search_args)
            return retriever.invoke(input=query)
        except pymilvus.exceptions.MilvusException as e:
            # params error code 1100
            if e.code == 1100:
                return self.retrieve(query, '', top_k, ef)
        except:
            # catch all exceptions
            return []

    async def aretrieve(self, query: str, expr: str, top_k: int = 20, ef: int = 300) -> List[Document]:
        """
        retrieval from vector store
        :param query: query
        :param expr: expr
        :param top_k: top k
        :param ef: ef
        :return: list of dict
        """
        if not query:
            raise ValueError("query is empty")
        try:
            search_args = self.search_args | {
                "expr": expr,
                "k": top_k,
                "param": {
                    "ef": ef
                }
            }
            retriever: VectorStoreRetriever = self.vector_store.as_retriever(search_kwargs=search_args)
            return await retriever.ainvoke(input=query)
        except pymilvus.exceptions.MilvusException as e:
            # params error code 1100
            if e.code == 1100:
                return await self.aretrieve(query, '', top_k, ef)
        except:
            # catch all exceptions
            return []

    def filter_and_rerank(self, docs: List[Document]) -> List[Document]:
        """
        filter and rerank
        :param docs: list of dict
        :return: list of dict
        """
        ranked_docs = reciprocal_rank_fusion([docs, ])
        #ranked_docs = remove_duplicated_documents(docs)
        final_docs = []
        # get poem detail from poem_store by doc.metadata['pid']
        for doc in ranked_docs:
            poem_detail: dict = self.poems_store.get(doc.metadata['pid'], {})
            if not poem_detail:
                continue
            new_document = Document(page_content=build_document_content_from_poem_details(poem_detail),
                                    metadata=poem_detail)
            final_docs.append(new_document)
        return final_docs

    def build_context(self, docs: List[Document]) -> str:
        """build context with poem document
        :param docs: documents
        """
        context = []
        total_context_chars = 0
        for doc in docs:
            if total_context_chars + len(doc.page_content) > self.max_token_chars:
                break
            context.append(doc.page_content)
            total_context_chars += len(doc.page_content)
        return '\n'.join(context)

    def generate(self, question: str, context: str = None) -> str:
        """
        generate answer
        :param question: question
        :param context: context
        :return: answer
        """
        try:
            if context:
                prompt = self.answer_prompt_template
            else:
                prompt = self.answer_without_context_prompt_template
            chain = prompt | self.llm_for_answer | StrOutputParser()
            return chain.invoke({"question": question, "context": context})
        except:
            # catch all exceptions
            return "对不起 我现在无法回答这个问题"

    async def agenerate(self, question: str, context: str = None) -> str:
        """
        generate answer
        :param question: question
        :param context: context
        :return: answer
        """
        try:
            if context:
                prompt = self.answer_prompt_template
            else:
                prompt = self.answer_without_context_prompt_template
            chain = prompt | self.llm_for_answer | StrOutputParser()
            return await chain.ainvoke({"question": question, "context": context})
        except:
            # catch all exceptions
            return "对不起 我现在无法回答这个问题"

    def stream_generate(self, question: str, context: str = None) -> Iterator[str]:
        """
        generate answer
        :param question: question
        :param context: context
        :return: answer
        """
        try:
            if context:
                prompt = self.answer_prompt_template
            else:
                prompt = self.answer_without_context_prompt_template
            chain = prompt | self.llm_for_answer | StrOutputParser()
            for chunk in chain.stream({"question": question, "context": context}):
                yield chunk
        except:
            # catch all exceptions
            yield "对不起 我现在无法回答这个问题"

    async def astream_generate(self, question: str, context: str = None) -> AsyncIterator[str]:
        """
        generate answer
        :param question: question
        :param context: context
        :return: answer
        """
        try:
            if context:
                prompt = self.answer_prompt_template
            else:
                prompt = self.answer_without_context_prompt_template
            chain = prompt | self.llm_for_answer | StrOutputParser()
            async for chunk in chain.astream({"question": question, "context": context}):
                yield chunk
        except:
            # catch all exceptions
            yield "对不起 我现在无法回答这个问题"

    def invoke(self, question: str, **kwargs) -> str:
        # rewrite
        query_dict = self.query_rewrite(question)
        # retrieve
        docs = self.retrieve(query_dict['query'], query_dict['expr'])
        # filter and rerank
        docs = self.filter_and_rerank(docs)
        # build context
        context = self.build_context(docs)
        # generate answer
        answer = self.generate(question, context)
        return answer

    async def ainvoke(self, question: str, **kwargs) -> str:
        query_dict = await self.aquery_rewrite(question)
        # retrieve
        docs = await self.aretrieve(query_dict['query'], query_dict['expr'])
        # filter and rerank
        docs = self.filter_and_rerank(docs)
        # build context
        context = self.build_context(docs)
        # generate answer
        answer = await self.agenerate(question, context)
        return answer

    def stream(self, question: str, **kwargs) -> Iterator[str]:
        # rewrite
        query_dict = self.query_rewrite(question)
        # retrieve
        docs = self.retrieve(query_dict['query'], query_dict['expr'])
        # filter and rerank
        docs = self.filter_and_rerank(docs)
        # build context
        context = self.build_context(docs)
        # generate answer
        for chunk in self.stream_generate(question, context):
            yield chunk

    async def astream(self, question: str, **kwargs) -> AsyncIterator[str]:
        query_dict = await self.aquery_rewrite(question)
        # retrieve
        docs = await self.aretrieve(query_dict['query'], query_dict['expr'])
        # filter and rerank
        docs = self.filter_and_rerank(docs)
        # build context
        context = self.build_context(docs)
        # generate answer
        async for chunk in self.astream_generate(question, context):
            yield chunk


def create_instance_by_qianfan_cloud(configs: dict) -> RagTangPoems:
    """百度千帆大模型技术栈"""

    import os
    import json
    from langchain_community.embeddings import QianfanEmbeddingsEndpoint
    from langchain_community.llms import QianfanLLMEndpoint
    from langchain_milvus import Milvus
    # set qianfan ak sk
    os.environ['QIANFAN_AK'] = configs['qianfan']['ak']
    os.environ['QIANFAN_SK'] = configs['qianfan']['sk']
    # init embedding
    embedding = QianfanEmbeddingsEndpoint(**configs['qianfan']['embedding'])
    # init llm
    llm = QianfanLLMEndpoint(**configs['qianfan']['rewrite_llm'])
    llm_for_answer = QianfanLLMEndpoint(**configs['qianfan']['answer_llm'])
    #  Milvus
    vector_store = Milvus(embedding_function=embedding, **configs['milvus']['kwargs'])
    # poem store
    poem_store = dict()
    with open(configs['poems_embedding_file'], 'r', encoding='utf8') as f:
        poems = json.load(f)
        for poem in poems:
            poem_store[poem['pid']] = poem
    # create instance
    return RagTangPoems(llm=llm, llm_for_answer=llm_for_answer, vector_store=vector_store,
                        poems_store=poem_store, max_token_chars=configs['max_token_chars_for_context'])
