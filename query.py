# -------------------- Imports --------------------
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


# -------------------- Structured Output Schema --------------------

class ResearchAnswer(BaseModel):
    answerr: str = Field(description="Direct answer to the research question")
    key_points: list[str] = Field(description="Key supporting points")
    evidence: dict[str, str] = Field(
        description="Paper-wise evidence with paper names as keys"
    )
    limitations: str = Field(description="Limitations or uncertainties")
    references: list[str] = Field(description="List of cited papers")


parser = PydanticOutputParser(pydantic_object=ResearchAnswer)


# -------------------- Prompt Template --------------------

prompt = PromptTemplate(
    template="""
You are an AI research assistant.

Your task is to answer the user’s question using ONLY the information provided
in the Context below.

STRICT CONSTRAINTS:
- Do NOT use any external knowledge.
- Do NOT guess, assume, or hallucinate.
- Every factual claim MUST be supported by the Context.
- If the Context does not contain enough information, respond EXACTLY with:
  "The provided documents do not contain sufficient information."

You MUST return your answer in a STRUCTURED JSON format that strictly follows
the given schema. Do not add any extra text outside the JSON.

Context:
{context}

Question:
{question}

STRUCTURED RESPONSE REQUIREMENTS:

- answerr:
  A concise, direct answer to the question based strictly on the Context.

- key_points:
  A list of clear bullet points summarizing the main findings or arguments
  directly supported by the Context.

- evidence:
  A dictionary where:
    - Each key is the name of a paper or source mentioned in the Context.
    - Each value is a short explanation of the evidence taken from that paper.

- limitations:
  A brief description of uncertainties, gaps, or limitations due to missing
  or incomplete information in the Context.

- references:
  A list of paper names or sources explicitly mentioned in the Context.

You MUST format your response exactly as shown below:
{format_instructions}
""",
    input_variables=["context", "question"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)


# -------------------- Load LLM (Mistral via Ollama) --------------------

llm = ChatOllama(
    model="mistral",
    temperature=0
)

# -------------------- Load Chroma Vector DB --------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

research_vector_db = Chroma(
    persist_directory="./chroma_db",
    collection_name="research_papers",
    embedding_function=embedding_model
)


# -------------------- Query Pipeline --------------------

def run_query(question: str) -> ResearchAnswer:
    """
    Executes a RAG-based query over the research corpus and
    returns a structured ResearchAnswer.
    """

    docs = research_vector_db.similarity_search(question, k=4)

    context_text = "\n\n".join(
        f"[Source: {doc.metadata.get('paper_name', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    )

    final_prompt = prompt.format(
        context=context_text,
        question=question
    )

    response = llm.invoke(final_prompt)
    raw_output = response.content.strip()

    return parser.parse(raw_output)


if __name__ == "__main__":
    question = input("Enter your research question: ")
    result = run_query(question)

    print("\n===== STRUCTURED RESEARCH ANSWER =====\n")
    print(result)

