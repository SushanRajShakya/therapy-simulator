import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from server.config import *
from server.rag_engine import RAGEngine

# Configure LangChain LLM
llm_model = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=MODEL_CONFIG["model"],
    temperature=MODEL_CONFIG["temperature"],
)

# Initialize RAG Engine
rag_engine = RAGEngine()


# Create CBT Sequential Chain with RAG Integration
def create_cbt_sequential_chain():
    # RAG Retrieval Function
    def retrieve_context(inputs):
        message = inputs["message"]
        # Retrieve relevant CBT knowledge and therapeutic context
        context = rag_engine.retrieve_context(message, k=3)
        return {"message": message, "retrieved_context": "\n\n".join(context)}

    # Step 1: Initial Assessment and Validation with RAG Context
    assessment_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist. Your task is to provide initial assessment and sentiment analysis of the patient.

        Patient message: {message}

        Relevant CBT knowledge and therapeutic context:
        {retrieved_context}

        Based on the patient's message and the provided therapeutic context, identify:
        1. The emotional state expressed
        2. Any cognitive patterns or beliefs mentioned
        3. Behavioral aspects described
        4. Which CBT techniques from the context might be most relevant

        Provide key points summarizing the patient's emotional state, cognitive patterns, behaviors, and suggest preliminary CBT approaches based on the retrieved knowledge.
        """,
    )

    # Step 2: CBT Technique Application with RAG Context
    technique_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist applying evidence-based techniques to help the patient whose emotional state is described in the assessment enclosed by ###.

        Assessment: ###{assessment}###

        Relevant CBT knowledge and context:
        {retrieved_context}

        Based on the assessment and the retrieved CBT knowledge, choose 2 or more relevant CBT techniques and suggest which techniques to focus on. Use the specific information from the retrieved context to inform your technique selection and application.
        
        Apply evidence-based CBT techniques from the retrieved knowledge, focusing on those most appropriate for the patient's current emotional state and cognitive patterns.
        """,
    )

    # Step 3: Action Planning and Follow-up with Context
    action_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist creating a therapeutic response based on CBT techniques suggested and the retrieved therapeutic knowledge.

        CBT Techniques suggested based on assessment: {techniques_application}

        Relevant therapeutic context and knowledge:
        {retrieved_context}

        Provide a final therapeutic response which always follows these guidelines along with the above steps:
        - Always respond as a therapist, never break character
        - Always maintain a professional, conversational therapeutic tone
        - Never provide medical advice or diagnoses
        - If crisis/self-harm is mentioned, acknowledge seriously and suggest professional help
        - Your final response should be natural and conversational, as if spoken directly to a patient as client
        - Do not include step-by-step breakdown in your final response
        - Provides specific, actionable next steps based on the retrieved knowledge
        - Be very kind, supportive, and empathetic in your final response
        - Integrate insights from the retrieved context naturally into your response
        - Do not directly use the CBT technique names in your final response, instead integrate them naturally into the conversation
        - Always end with an open-ended question to encourage further discussion
        """,
    )

    # Create the chain with RAG integration
    context_retrieval = RunnableLambda(retrieve_context)

    assessment = assessment_prompt | llm_model

    apply_cbt_technique = technique_prompt | llm_model

    therapeutic_response = action_prompt | llm_model | StrOutputParser()

    return RunnableSequence(
        context_retrieval, assessment, apply_cbt_technique, therapeutic_response
    )
