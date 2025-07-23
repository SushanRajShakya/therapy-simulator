import os
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from models.chat import *
from constants import *

# Load environment variables
load_dotenv(override=True)

app = FastAPI(title=SERVER_NAME)

# Configure OpenAI
open_ai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=open_ai_api_key)


@app.post("/chat", response_model=ChatResponse)
def chat_with_llm(request: ChatRequest):
    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful therapy assistant trained in CBT techniques.",
                },
                {"role": "user", "content": request.message},
            ],
            temperature=1,
        )

        llm_response = response.choices[0].message.content

        data = ChatResponse(response=llm_response, session_id=request.session_id)

        print(f"{str(data)}")

        return data
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")  # Add logging
        raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": SERVER_NAME}
