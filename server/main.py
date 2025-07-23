import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_openai import OpenAI as LangChainOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

from chat_model import *
from constants import *
from config import CBT_THERAPIST_SYSTEM_PROMPT, MODEL_CONFIG

# Load environment variables
load_dotenv(override=True)

app = FastAPI(title=SERVER_NAME)

# Configure OpenAI
open_ai_api_key = os.getenv("OPENAI_API_KEY")

# Configure LangChain LLM
langchain_llm = LangChainOpenAI(
    api_key=open_ai_api_key,
    model=MODEL_CONFIG["model"],
    temperature=MODEL_CONFIG["temperature"],
)


# Create CBT Sequential Chain
def create_cbt_sequential_chain():
    # Step 1: Initial Assessment and Validation
    assessment_prompt = PromptTemplate(
        input_variables=["user_message"],
        template="""
        {cbt_system_prompt}
        
        STEP 1: INITIAL ASSESSMENT AND VALIDATION
        
        User message: {user_message}
        
        First, acknowledge and validate the user's feelings and experience. Then identify:
        1. The emotional state expressed
        2. Any cognitive patterns or beliefs mentioned
        3. Behavioral aspects described
        
        Provide a warm, validating response that shows you understand their experience.
        
        Assessment:
        """.replace(
            "{cbt_system_prompt}", CBT_THERAPIST_SYSTEM_PROMPT
        ),
    )

    # Step 2: CBT Technique Application
    technique_prompt = PromptTemplate(
        input_variables=["user_message", "assessment"],
        template="""
        Based on the initial assessment: {assessment}
        
        STEP 2: CBT TECHNIQUE APPLICATION
        
        Original user message: {user_message}
        
        Now apply appropriate CBT techniques from the following list:
        - ABC Model (Activating Event, Belief, Consequences)
        - Guided Discovery through questions
        - Cognitive Restructuring
        - Behavioral Experiments
        - Activity Scheduling
        - Journaling suggestions
        - Role-Playing scenarios
        
        Choose 1-2 most relevant techniques and guide the user through them with specific questions or exercises.
        
        CBT Technique Application:
        """,
    )

    # Step 3: Action Planning and Follow-up
    action_prompt = PromptTemplate(
        input_variables=["user_message", "assessment", "technique_application"],
        template="""
        User message: {user_message}
        Assessment: {assessment}
        Technique applied: {technique_application}
        
        STEP 3: ACTION PLANNING AND THERAPEUTIC RESPONSE
        
        Now create a complete therapeutic response that:
        1. Integrates the validation from Step 1
        2. Incorporates the CBT technique from Step 2
        3. Provides specific, actionable next steps
        4. Asks open-ended questions for continued exploration
        5. Maintains professional, conversational therapeutic tone
        
        IMPORTANT: Your final response should be natural and conversational, as if spoken directly to the client. Do not include the step-by-step breakdown in your final response.
        
        Final Therapeutic Response:
        """,
    )

    # Create individual chains
    assessment_chain = LLMChain(
        llm=langchain_llm, prompt=assessment_prompt, output_key="assessment"
    )

    technique_chain = LLMChain(
        llm=langchain_llm, prompt=technique_prompt, output_key="technique_application"
    )

    action_chain = LLMChain(
        llm=langchain_llm, prompt=action_prompt, output_key="final_response"
    )

    # Create sequential chain
    sequential_chain = SequentialChain(
        chains=[assessment_chain, technique_chain, action_chain],
        input_variables=["user_message"],
        output_variables=["final_response"],
        verbose=True,
    )

    return sequential_chain


# Initialize the CBT sequential chain
cbt_chain = create_cbt_sequential_chain()


@app.post("/chat", response_model=ChatResponse)
def chat_with_llm(request: ChatRequest):
    try:
        # Use the CBT sequential chain
        result = cbt_chain.run(user_message=request.message)

        # The sequential chain returns the final therapeutic response
        llm_response = result

        data = ChatResponse(response=llm_response, session_id=request.session_id)

        print(f"{str(data)}")

        return data
    except Exception as e:
        print(f"CBT Chain error: {str(e)}")  # Add logging
        raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": SERVER_NAME}
