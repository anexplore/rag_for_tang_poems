# -*- coding: utf-8 -*-
"""
api server base on fastapi
/qa_poems
"""
import logging
import json
import traceback

import fastapi
import uvicorn
from fastapi.responses import Response, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag_tang_poems import create_instance_by_qianfan_cloud, RagTangPoems


class QARequest(BaseModel):
    question: str = Field(description='question to ask')
    stream: bool = Field(default=False, description='streaming response')


def load_config(file='config.json'):
    """
    load config from file
    """
    with open(file, 'r', encoding='utf8') as f:
        return json.load(f)


config = load_config()
rag: RagTangPoems = create_instance_by_qianfan_cloud(config)
app = fastapi.FastAPI()


@app.post("/qa_poems", response_class=Response)
async def qa_poems(question: QARequest):
    """
    qa poems
    """
    if not question.stream:
        try:
            answer = await rag.ainvoke(question.question)
        except:
            logging.error('process question failed: %s, %s' % (question.question, traceback.format_exc()))
            answer = '我现在无法回答这个问题'
        return JSONResponse(content={'content': answer}, media_type='application/json')
    else:
        async def stream():
            try:
                async for chunk in rag.astream(question.question):
                    if not chunk:
                        yield 'data: {"content": "", "finished": true, "finish_reason": "stop"}\n\n'
                        return
                    data = json.dumps({'content': chunk}, ensure_ascii=False)
                    yield 'data: %s\n\n' % data
            except:
                logging.error('process question failed: %s, %s' % (question.question, traceback.format_exc()))
                yield 'data: {"content": "\n异常中断\n", "finish": true, "finish_reason": "exception"}\n\n'
                return
        return StreamingResponse(stream(), media_type='text/event-stream')


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8888, limit_concurrency=10)