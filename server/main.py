from fastapi import FastAPI, HTTPException

from server.chat_model import *
from server.constants import *
from server.cbt_chain import create_cbt_sequential_chain
from server.session_manager import session_manager

app = FastAPI(title=SERVER_NAME)


# Initialize the CBT sequential chain
cbt_chain = create_cbt_sequential_chain()


@app.post("/chat", response_model=ChatResponse)
def chat_with_llm(request: ChatRequest):
    try:
        # Handle session ending request
        if request.end_session:
            # Generate final conclusion
            conclusion = session_manager.generate_session_conclusion(request.session_id)

            # Add the final user message and conclusion to session
            session_manager.add_message(request.session_id, "user", request.message)
            session_manager.add_message(request.session_id, "assistant", conclusion)

            return ChatResponse(
                response=conclusion,
                session_id=request.session_id,
                is_session_ended=True,
            )

        # Add user message to session
        session_manager.add_message(request.session_id, "user", request.message)

        # Get conversation context for more cost-effective processing
        conversation_context = session_manager.get_conversation_context(
            request.session_id
        )

        # Use the CBT sequential chain with conversation context
        llm_response = cbt_chain.invoke(
            {"message": request.message, "conversation_context": conversation_context}
        )

        # Add assistant response to session
        session_manager.add_message(request.session_id, "assistant", llm_response)

        return ChatResponse(response=llm_response, session_id=request.session_id)

    except Exception as e:
        print(f"CBT Chain error: {str(e)}")  # Add logging
        raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": SERVER_NAME}
