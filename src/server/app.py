"""
API
"""
import logging
from io import BytesIO

import fitz
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from store import ElasticsearchDataStore
import configuration
import settings
from .models import Question, Answer, Result, Metadata

app = FastAPI()

store = ElasticsearchDataStore()


@app.post("/ask/")
async def ask(question: Question) -> Answer:
    """
    Ask a question.
    """
    ask_llm = settings.CONFIGURATION.llm == configuration.LargeLanguageModel.OPENAI
    results = store.search(question.question)

    if ask_llm:
        answer = store.ask(question.question)
    else:
        answer = "Cannot provide an answer as LLM is not configured."

    # NOTE: this indirection is needed until Langchain upgrades to pydantic v2.
    # At that point the Answer model could just have a list[Document] as the
    # resul and Metatada / Result could be removed unless needed to inject extra
    # information.
    mapped_results = [
        Result(
            page_content=r[0].page_content,
            metadata=Metadata(page=r[0].metadata.get("page", 0), source=r[0].metadata["source"]),
            score=r[1]
        ) for r in results
    ]
    return Answer(answer=answer, results=mapped_results)


@app.get("/pdfpage")
async def pdfloc(filename: str, page: int) -> None:
    """
    Extract a PDF page.
    """
    try:
        doc = fitz.open(settings.DOCS_DIR / filename)  # open document
    except Exception as ex:
        logging.error(ex)
        raise HTTPException(status_code=404, detail="File not found") from ex
    try:
        page = doc.load_page(page)
        pixmap = page.get_pixmap()
        buf = BytesIO(pixmap.pil_tobytes("jpeg"))
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="image/jpeg",
            headers={"Content-Disposition": 'inline; filename="page.jpg"'})
    except Exception as ex:
        logging.error(ex)
        raise HTTPException(status_code=404, detail="Page not found or not loadable") from ex
