from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.similarity_engine import compute_similarity_score

app = FastAPI(title="AI Influence Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    student: str
    llms: list[str]

@app.post("/analyze")
def analyze(data: AnalyzeRequest):
    try:
        score = compute_similarity_score(data.student, data.llms)
        return {"similarity_score": round(score, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))