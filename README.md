# rag_for_tang_poems
唐诗三百首RAG(检索增强生成)

## 1、诗词范围

[唐诗300首](tangshi.md)

后续如果有时间可以继续补充宋词等

## 2、向量数据库

### 2.1 向量的生成

使用百度千帆提供的embedding接口


[split_and_embedding.py](split_and_embedding.py)

### 2.2 向量的索引

使用Milvus作为向量数据库

- 这里将向量化的诗词数据写入到了文件中，以便于在使用不同向量库结构时无需重新embedding
- 使用[milvus_collection.py](milvus_collection.py)将向量数据写入到Milvus中

### 2.3 说明

诗包含的字段有标题、作者、诗文、译文、注解和赏析几部分；

- 将标题和作者联合embedding
- 将作者作为数据库的标量字段
- 将诗文、译文和赏析先按照\n换行符切分为行，然后按照最大352长度字符对行进行合并。
如果单行的长度大于352，则对单行进行步长为352字符、重叠30字符的滑动窗口进行切分


## 3、RAG细节

### 3.1 RAG过程

Query改写和构建 -> 检索 -> rerank -> 生成

### 3.2 相关Prompt

- query的改写和milvus expr条件的生成在一个Prompt中
- 结果的生成使用两个Prompt，当检索的结果不足时，使用第二个Prompt直接使用LLM原生能力回答问题
- Prompt中上下文context的构建：放入诗词的标题、作者、正文、译文和赏析字段。
**注意这里会比较长，可以控制检索top-k或者通过首先调用大模型来根据问题决定context需要放置哪些内容**

### 3.3 模型选择

- query的改写使用效果较好的ernie-3.5/ernie-4
- 答案的生成使用支持更大上下文的免费的ERNIE-Speed-128K

## 4、接口调用

### 4.1 接口

```text
POST /qa_poems
{
    "question": "李白的诗",
    "stream": false
}
```

## 5、部署与启动

- milvus 部署参考[milvuswiki](https://milvus.io/docs/install_standalone-docker.md)
- 依赖包括langchain\fastapi\unicorn\qianfan等
- 开通[百度千帆](https://qianfan.cloud.baidu.com/)的接口

```text
python serve.py
```

## 6、TODO

- [ ] 1、增加更多的诗词范围
- [ ] 2、优化向量数据库索引结构
- [ ] 3、Query改写、答案生成等Prompt的优化
- [ ] 4、根据问题动态调整context
