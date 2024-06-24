# -*- coding: utf-8 -*-
"""
创建Milvus Collection

Collection结构
 - pid: int 诗歌ID
 - author: str 作者
 - vector: list[float] 向量
 - text: str 向量对应内容
 - pk, auto_id = True
"""
import json

from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema, utility


def create_collection_and_index(connection_args: dict, data_file='tang_poems_embedded.json'):
    """
    创建Milvus Collection
    :param connection_args: dict Milvus连接参数
    :param data_file: str 数据文件
    """
    connections.connect(**connection_args)
    collection_name = 'poem_collection'
    # drop
    if utility.has_collection(collection_name):
        Collection(name=collection_name).drop()
    fields = [
        FieldSchema(name="pk", is_primary=True, dtype=DataType.INT64, auto_id=True),
        FieldSchema(name="pid", dtype=DataType.INT32),
        FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
    ]
    schema = CollectionSchema(fields=fields, description="poem collection")
    collection = Collection(name=collection_name, schema=schema)

    # 插入数据
    with open(data_file, 'r', encoding='utf8') as f:
        poems = json.load(f)
    for poem in poems:
        # 将 title + author 插入
        data = [{
            "pid": poem['pid'],
            "author": poem['author'],
            "vector": poem['title_and_author_embedding'],
            "text": poem['title'] + ' ' + poem['author']
        }]
        # 将 content chunks translation chunks appreciation chunks 插入
        for field in ['content', 'translation', 'appreciation']:
            chunk_key = '%s_chunks' % field
            embedding_key = '%s_chunks_embedding' % field
            for chunk, embedding in zip(poem[chunk_key], poem[embedding_key]):
                data.append({
                    "pid": poem['pid'],
                    "author": poem['author'],
                    "vector": embedding,
                    "text": chunk
                })
        res = collection.insert(data)
        print('insert %s chunk' % res.insert_count)
    collection.flush()
    # 创建索引
    index_params = {
        "index_type": "HNSW",
        "params": {
            "M": 32,
            "efConstruction": 200
        },
        "metric_type": "L2"
    }
    collection.create_index(field_name="vector", index_params=index_params)
    connections.disconnect(alias='default')


if __name__ == '__main__':
    connection_args = {
        "alias": 'default',
        "uri": "tcp://host:port"
    }
    create_collection_and_index(connection_args)