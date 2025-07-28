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
        You are a professional CBT therapist conducting an initial assessment. Your task is to analyze the patient's input and provide a structured assessment.

        Patient message: {message}

        Relevant therapeutic examples and CBT knowledge:
        {retrieved_context}

        Based on the patient's message and the retrieved therapeutic knowledge, provide a comprehensive assessment covering:

        1. EMOTIONAL STATE: What emotions are being expressed (anxiety, depression, anger, etc.)?
        2. COGNITIVE PATTERNS: What thought patterns, beliefs, or cognitive distortions are evident?
        3. BEHAVIORAL ASPECTS: What behaviors or avoidance patterns are described?
        4. TRIGGERS & CONTEXT: What situations or events seem to trigger these responses?
        5. SEVERITY & IMPACT: How significantly is this affecting their daily functioning?

        Reference similar cases from the therapeutic examples when relevant. Keep your assessment clinical but empathetic.
        Focus on identifying patterns that align with CBT principles and the knowledge retrieved from the context.
        
        Format your response as a structured assessment that will inform CBT technique selection.
        """,
    )

    # Step 2: CBT Technique Application with RAG Context
    technique_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist selecting and planning evidence-based interventions.

        Assessment: ###{assessment}###

        Relevant therapeutic examples and CBT knowledge:
        {retrieved_context}

        Based on the assessment and the retrieved therapeutic examples and CBT knowledge:

        1. TECHNIQUE SELECTION: Identify 2-3 most appropriate CBT techniques for this specific case from the retrieved knowledge (e.g., cognitive restructuring, behavioral activation, exposure therapy, mindfulness, ABC model, problem-solving)

        2. EVIDENCE-BASED RATIONALE: Explain why these techniques are suitable based on:
           - The identified cognitive patterns and distortions
           - The emotional state and behavioral patterns
           - Similar successful applications from the therapy examples

        3. APPLICATION STRATEGY: Detail how each technique should be adapted to this patient's specific situation, referencing similar cases from the retrieved examples

        4. THERAPEUTIC APPROACH: Consider the communication style and intervention methods demonstrated in the professional therapy examples

        Use the retrieved CBT knowledge to ensure techniques are applied correctly and reference successful therapeutic interactions from the examples to inform your approach.
        
        Provide a clear, structured plan that will guide the final therapeutic response.
        """,
    )

    # Step 3: Rich Context Response Generation
    action_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist creating a compassionate, evidence-based therapeutic response.

        Original patient message: {message}

        Assessment findings: ###{assessment}###

        CBT technique recommendations: ###{techniques_application}###

        Relevant therapeutic examples and knowledge:
        {retrieved_context}

        Create a rich, contextual therapeutic response that:

        THERAPEUTIC COMMUNICATION:
        - Model your tone and style on the professional examples in the retrieved context
        - Respond as if speaking directly to the patient with warmth and understanding
        - Acknowledge and validate their feelings and experiences
        - Never provide medical diagnoses or advice

        INTEGRATION OF CBT TECHNIQUES:
        - Seamlessly weave the recommended techniques into natural conversation
        - Don't explicitly name techniques - integrate them organically
        - Use language and approaches demonstrated in the retrieved therapy examples
        - Reference similar situations from the context when relevant

        ACTIONABLE GUIDANCE:
        - Provide specific, manageable next steps based on the technique recommendations
        - Draw from successful interventions shown in the therapy examples
        - Offer practical tools or exercises that align with the assessment findings
        - Make suggestions feel collaborative rather than prescriptive

        CONVERSATIONAL FLOW:
        - End with an open-ended question that encourages further exploration
        - Maintain hope and emphasize the patient's strengths and agency
        - Keep the response conversational and accessible, not clinical
        - Show empathy while gently introducing therapeutic perspectives

        Use the retrieved context to inform your communication style and ensure your response reflects evidence-based therapeutic practice.
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
