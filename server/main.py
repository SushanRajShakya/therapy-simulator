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

        # Add user message to session first (always track what user says)
        session_manager.add_message(request.session_id, "user", request.message)

        # Classify the message to determine response strategy
        message_classification = session_manager.classify_message(
            request.message, request.session_id
        )
        print(
            f"Message classification: {message_classification} for message: '{request.message[:50]}...'"
        )

        # Handle session end detection
        if message_classification == "SESSION_END":
            print("Natural session end detected")
            # Generate final conclusion
            conclusion = session_manager.generate_session_conclusion(request.session_id)
            session_manager.add_message(request.session_id, "assistant", conclusion)

            return ChatResponse(
                response=conclusion,
                session_id=request.session_id,
                is_session_ended=True,
            )

        # For simple messages, use lightweight response (no RAG/CBT chain)
        if message_classification in ["GREETING", "PROCEDURAL", "SMALL_TALK"]:
            print(f"Using simple response for {message_classification}")
            simple_response = session_manager.generate_simple_response(
                request.message, request.session_id, message_classification
            )
            session_manager.add_message(
                request.session_id, "assistant", simple_response
            )

            return ChatResponse(response=simple_response, session_id=request.session_id)

        # For therapeutic content, use full CBT chain with RAG
        if message_classification == "THERAPEUTIC":
            print("Using full CBT chain with RAG")
            # Get conversation context for more cost-effective processing
            conversation_context = session_manager.get_conversation_context(
                request.session_id
            )

            # Use the CBT sequential chain with conversation context
            llm_response = cbt_chain.invoke(
                {
                    "message": request.message,
                    "conversation_context": conversation_context,
                }
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
