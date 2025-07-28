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
    # Enhanced RAG Retrieval Function
    def retrieve_context(inputs):
        message = inputs["message"]

        # Retrieve relevant therapy conversations and CBT knowledge
        # Get more context pieces for better therapeutic guidance
        context_docs = rag_engine.retrieve_context(message, k=4)

        # Format the context for better readability
        formatted_context = []
        for i, context in enumerate(context_docs):
            # Check if it's a conversation format
            if "Client:" in context and "Therapist:" in context:
                formatted_context.append(
                    f"Example Therapy Interaction {i+1}:\n{context}"
                )
            else:
                formatted_context.append(f"Therapeutic Knowledge {i+1}:\n{context}")

        return {"message": message, "retrieved_context": "\n\n".join(formatted_context)}

    # Step 1: Initial Assessment and Validation with RAG Context
    assessment_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist. Your task is to provide initial assessment and sentiment analysis of the patient.

        Patient message: {message}

        Relevant therapeutic examples and CBT knowledge:
        {retrieved_context}

        Based on the patient's message and the provided therapeutic examples and knowledge, identify:
        1. The emotional state expressed
        2. Any cognitive patterns or beliefs mentioned
        3. Behavioral aspects described
        4. Similar therapeutic situations from the examples that might inform your approach
        5. Which CBT techniques from the context might be most relevant

        Use the therapy conversation examples to understand how similar client concerns have been addressed professionally.
        Provide key points summarizing the patient's emotional state, cognitive patterns, behaviors, and suggest preliminary CBT approaches based on the retrieved knowledge and examples.
        """,
    )

    # Step 2: CBT Technique Application with RAG Context
    technique_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist applying evidence-based techniques to help the patient whose emotional state is described in the assessment enclosed by ###.

        Assessment: ###{assessment}###

        Relevant therapeutic examples and CBT knowledge:
        {retrieved_context}

        Based on the assessment and the retrieved therapeutic examples and CBT knowledge:
        1. Identify 2-3 most relevant CBT techniques for this specific case
        2. Reference similar cases from the therapy conversation examples to inform your approach
        3. Explain how these techniques should be applied to the patient's specific situation
        4. Consider the therapeutic communication style demonstrated in the examples

        Use the professional therapy conversation examples as models for effective therapeutic communication and intervention strategies.
        Choose techniques that have been demonstrated to work well for similar client presentations in the retrieved examples.
        """,
    )

    # Step 3: Action Planning and Follow-up with Context
    action_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist creating a therapeutic response based on CBT techniques suggested and the retrieved therapeutic knowledge.

        CBT Techniques suggested based on assessment: {techniques_application}

        Relevant therapeutic examples and knowledge:
        {retrieved_context}

        Create a final therapeutic response following these guidelines:
        - Model your communication style on the professional therapy examples provided
        - Always respond as a therapist, never break character
        - Maintain a professional, conversational therapeutic tone similar to the examples
        - Never provide medical advice or diagnoses
        - If crisis/self-harm is mentioned, acknowledge seriously and suggest professional help
        - Your response should be natural and conversational, as if spoken directly to a patient
        - Do not include step-by-step breakdown in your final response
        - Provide specific, actionable next steps informed by the retrieved therapeutic examples
        - Be very kind, supportive, and empathetic in your final response
        - Integrate insights from the therapy conversation examples naturally
        - Use the communication patterns demonstrated in the retrieved examples as a guide
        - Do not directly name CBT techniques, instead integrate them naturally into the conversation
        - Always end with an open-ended question to encourage further discussion, similar to the examples

        Draw inspiration from the professional therapeutic responses in the retrieved examples to create an authentic, helpful response.
        """,
    )

    # Create the chain with RAG integration
    context_retrieval = RunnableLambda(retrieve_context)

    # Step 1: Assessment with context
    def run_assessment(inputs):
        assessment_result = (assessment_prompt | llm_model).invoke(inputs)
        return {
            "message": inputs["message"],
            "retrieved_context": inputs["retrieved_context"],
            "assessment": assessment_result.content,
        }

    # Step 2: Technique application with context
    def run_technique_application(inputs):
        technique_result = (technique_prompt | llm_model).invoke(inputs)
        return {
            "message": inputs["message"],
            "retrieved_context": inputs["retrieved_context"],
            "assessment": inputs["assessment"],
            "techniques_application": technique_result.content,
        }

    # Step 3: Final therapeutic response
    def run_therapeutic_response(inputs):
        response_result = (action_prompt | llm_model | StrOutputParser()).invoke(inputs)
        return response_result

    assessment_step = RunnableLambda(run_assessment)
    technique_step = RunnableLambda(run_technique_application)
    response_step = RunnableLambda(run_therapeutic_response)

    return RunnableSequence(
        context_retrieval, assessment_step, technique_step, response_step
    )
