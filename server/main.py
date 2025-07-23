from fastapi import FastAPI, HTTPException

from chat_model import *
from constants import *
from cbt_chain import create_cbt_sequential_chain

app = FastAPI(title=SERVER_NAME)


# Initialize the CBT sequential chain
cbt_chain = create_cbt_sequential_chain()


@app.post("/chat", response_model=ChatResponse)
def chat_with_llm(request: ChatRequest):
    try:
        # Use the CBT sequential chain
        llm_response = cbt_chain.run(user_message=request.message)

        data = ChatResponse(response=llm_response, session_id=request.session_id)

        print(f"{str(data)}")

        return data
    except Exception as e:
        print(f"CBT Chain error: {str(e)}")  # Add logging
        raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": SERVER_NAME}
