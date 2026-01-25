# server.py
import os
from dotenv import load_dotenv

load_dotenv()  # .env'den GROQ_API_KEY okur

from fastapi import FastAPI
from pydantic import BaseModel

# senin agent dosyan
from intent_parser_agent import intent_parser_agent

app = FastAPI(title="Intent Parser API", version="1.0")


class ParseRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"ok": True, "has_groq_key": bool(os.getenv("GROQ_API_KEY"))}


@app.post("/parse")
def parse(req: ParseRequest):
    # agno run çıktısı genelde out.content'te
    out = intent_parser_agent.run(req.text)
    return out.content  # schema objesi ise otomatik JSON döner