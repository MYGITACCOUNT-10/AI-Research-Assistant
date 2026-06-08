from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time
from query import run_query

app = FastAPI(
    title="AI Research Assistant API",
    description="RAG-based Q&A system serving structured research insights",
    version="1.0.0"
)

class ResearchRequest(BaseModel):
    question: str = Field(..., description="The research question to be answered", example="Explain CNN-based deepfake detection methods")

class EvidenceItem(BaseModel):
    paper_name: str
    explanation: str

class ResearchResponse(BaseModel):
    answerr: str = Field(description="Direct answer to the research question")
    key_points: list[str] = Field(description="Key supporting points")
    evidence: dict[str, str] = Field(description="Paper-wise evidence with paper names as keys")
    limitations: str = Field(description="Limitations or uncertainties")
    references: list[str] = Field(description="List of cited papers")
    latency_ms: float = Field(description="Processing latency in milliseconds")

@app.post("/api/v1/query", response_model=ResearchResponse)
def execute_query(request: ResearchRequest):
    start_time = time.time()
    try:
        # Execute RAG pipeline
        result = run_query(request.question)
        
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        return ResearchResponse(
            answerr=result.answerr,
            key_points=result.key_points,
            evidence=result.evidence,
            limitations=result.limitations,
            references=result.references,
            latency_ms=latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ai-research-assistant"}
