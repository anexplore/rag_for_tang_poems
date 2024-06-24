# -*- coding: utf-8 -*-
"""
Split Poems and Embed them
"""
import copy
import json
import time

from typing import List


def split_to_chunks(data: str, chunk_size:int = 512, overlap:int = 30) -> List[str]:
    """
    Split a string into chunks with a given size and overlap
    :param data: str input string
    :param chunk_size: int size of each chunk
    :param overlap: int overlap between chunks
    :return: list of str chunks
    """
    chunks = []
    for i in range(0, len(data), chunk_size-overlap):
        chunks.append(data[i:i+chunk_size])
    return chunks


def split_poems(poems: List[dict], max_chunk_size: int = 512) -> List[dict]:
    """
    poem结构:
    - title: str 诗名
    - content: str 内容
    - author: str 作者
    - dynasty: str 朝代
    - translation: str 译文 白话文
    - annotation: str 注解
    - appreciation: str 赏析

    需要被切分的字段：
    - content
    - translation
    - appreciation

    切片规则：
    - 按照段落切片并合并成chunk
    - 保证每个chunk的所有段落长度之和不超过max_chunk_size
    - 如果单个段落长度超过max_chunk_size，则将该段落切分成多个chunk
    :param poems: list[dict] 诗歌列表
    :dst_file: str 输出文件
    :param max_chunk_size: int 最大chunk长度
    :return: list[dict] 切片后的诗歌列表
    """
    chunked_poems = []
    for poem in poems:
        copied_poem = copy.deepcopy(poem)
        # 按照段落切分
        for field in ['content', 'translation', 'appreciation']:
            if field not in copied_poem:
                continue
            if not copied_poem[field]:
                copied_poem['%s_chunks' % field] = []
                continue
            chunks = []
            items = []
            items_total_length = 0
            paragraphs = [item.strip() for item in copied_poem[field].split('\n')]
            for paragraph in paragraphs:
                if not paragraph:
                    continue
                if items_total_length + len(paragraph) < max_chunk_size:
                    items.append(paragraph)
                    items_total_length += len(paragraph)
                else:
                    if items:
                        chunks.append(' '.join(items))
                    items = []
                    items_total_length = 0
                    if len(paragraph) > max_chunk_size:
                        chunks.extend(split_to_chunks(paragraph, chunk_size=max_chunk_size))
                    else:
                        items = [paragraph,]
                        items_total_length = len(paragraph)
            if items:
                chunks.append(' '.join(items))
            copied_poem['%s_chunks' % field] = chunks
        chunked_poems.append(copied_poem)
    return chunked_poems


def split_and_save_poem_fields(poems_file='tang_poems.json', dst_file='tang_poems_chunked.json', max_chunk_size=352):
    """
    读取诗歌文件，切分诗歌字段，保存切分后的诗歌
    :param poems_file: str 诗歌文件
    :param dst_file: str 输出文件
    :param max_chunk_size: int 最大chunk长度
    """
    with open(poems_file, 'r', encoding='utf8') as f:
        poems = json.load(f)

    chunked_poems = split_poems(poems, max_chunk_size=max_chunk_size)
    with open(dst_file, 'w', encoding='utf8') as f:
        json.dump(chunked_poems, f, ensure_ascii=False, indent=2)


def embed_poems(poems_file='tang_poems_chunked.json', dst_file='tang_poems_embedded.json', embeddings=None, qps=4):
    """
    读取切分后的诗歌文件，对诗歌字段进行embedding，保存embedding后的诗歌
    :param poems_file: str 切分后的诗歌文件
    :param dst_file: str 输出文件
    :param embeddings: embedding对象
    :param qps: int 每秒请求数
    """
    with open(poems_file, 'r', encoding='utf8') as f:
        poems = json.load(f)
    request_interval = 1.0 / qps
    for poem in poems:
        for field in ['content_chunks', 'translation_chunks', 'appreciation_chunks']:
            if field not in poem:
                continue
            if not poem[field]:
                poem['%s_embedding' % field] = []
                continue
            print('process %s' % poem[field])
            poem['%s_embedding' % field] = embeddings.embed_documents(poem[field])
            time.sleep(request_interval)
        poem['title_embedding'] = embeddings.embed_documents([poem['title']])[0]
        time.sleep(request_interval)
        poem['title_and_author_embedding'] = embeddings.embed_documents([poem['title'] + ' ' + poem['author']])[0]
        time.sleep(request_interval)
    with open(dst_file, 'w', encoding='utf8') as f:
        json.dump(poems, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # split to chunks
    #split_and_save_poem_fields()
    #from langchain_community.embeddings import QianfanEmbeddingsEndpoint
    #embeddings = QianfanEmbeddingsEndpoint(qianfan_ak='', qianfan_sk='')
    #embed_poems(embeddings=embeddings)
    pass