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
    # RAG Retrieval Function for Response Generation
    def retrieve_therapeutic_responses(inputs):
        message = inputs["message"]
        conversation_context = inputs.get("conversation_context", "")
        assessment = inputs["assessment"]
        techniques_application = inputs["techniques_application"]

        # Retrieve relevant therapist responses from the dataset
        # Use the specialized method to get actual therapeutic responses
        therapist_responses = rag_engine.retrieve_therapist_responses(message, k=4)

        # Format the responses for the prompt
        formatted_responses = []
        for i, response in enumerate(therapist_responses):
            formatted_responses.append(f"Example Response {i+1}: {response}")

        return {
            "message": message,
            "conversation_context": conversation_context,
            "assessment": assessment,
            "techniques_application": techniques_application,
            "retrieved_responses": "\n\n".join(formatted_responses),
        }

    # Step 1: Initial Assessment and Validation
    assessment_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist conducting an ongoing assessment. Your task is to analyze the patient's input within the context of your ongoing therapeutic relationship.

        Current patient message: {message}

        Conversation context (summary and recent history):
        {conversation_context}

        Based on the patient's current message and conversation history, provide a comprehensive assessment covering:

        1. CONVERSATION STATE: Determine if this is a first interaction or continuation of an ongoing therapeutic relationship
        2. EMOTIONAL STATE: What emotions are being expressed in this message and how do they relate to previous sessions?
        3. COGNITIVE PATTERNS: What thought patterns, beliefs, or cognitive distortions are evident? Any patterns from previous conversations?
        4. BEHAVIORAL ASPECTS: What behaviors or avoidance patterns are described? Any changes from earlier discussions?
        5. TRIGGERS & CONTEXT: What situations or events seem to trigger these responses? Connection to previous sessions?
        6. THERAPEUTIC PROGRESS: How does this message show progress or challenges compared to earlier conversations?
        7. SEVERITY & IMPACT: How significantly is this affecting their daily functioning?

        IMPORTANT: Pay special attention to the conversation context to determine:
        - Is this the patient's first message in the session?
        - Are they continuing a previous topic or introducing something new?
        - What therapeutic rapport has already been established?

        Keep your assessment clinical but empathetic.
        Consider the ongoing therapeutic relationship and build upon previous insights when available.
        
        Format your response as a structured assessment that will inform CBT technique selection and conversation continuity.
        """,
    )

    # Step 2: CBT Technique Application
    technique_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist selecting and planning evidence-based interventions.

        Assessment: ###{assessment}###

        Based on the assessment, identify appropriate CBT techniques:

        1. TECHNIQUE SELECTION: Identify 2-3 most appropriate CBT techniques for this specific case (e.g., cognitive restructuring, behavioral activation, exposure therapy, mindfulness, ABC model, problem-solving)

        2. EVIDENCE-BASED RATIONALE: Explain why these techniques are suitable based on:
           - The identified cognitive patterns and distortions
           - The emotional state and behavioral patterns
           - The client's specific situation and needs

        3. APPLICATION STRATEGY: Detail how each technique should be adapted to this patient's specific situation

        4. THERAPEUTIC APPROACH: Consider the most effective communication style and intervention methods for this client

        Provide a clear, structured plan that will guide the final therapeutic response.
        """,
    )

    # Step 3: Rich Context Response Generation with Retrieved Responses
    action_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist creating a compassionate, evidence-based therapeutic response.

        Original patient message: {message}

        Assessment findings: ###{assessment}###

        CBT technique recommendations: ###{techniques_application}###

        Conversation context (summary and recent history):
        {conversation_context}

        Example therapeutic responses from experienced therapists:
        {retrieved_responses}

        Create a rich, contextual therapeutic response that:

        CONVERSATION AWARENESS:
        - Review the conversation context to understand where you are in the therapeutic relationship
        - If this is a continuation of an ongoing conversation, respond naturally without greeting the patient again
        - Only provide initial greetings if this appears to be the very first interaction
        - Build upon previous topics and insights from the conversation history

        THERAPEUTIC COMMUNICATION:
        - Model your tone and style on the professional response examples provided
        - Respond as if speaking directly to the patient with warmth and understanding
        - Acknowledge and validate their feelings and experiences
        - Never provide medical diagnoses or advice
        - Maintain conversational continuity based on the session history

        INTEGRATION OF CBT TECHNIQUES:
        - Seamlessly weave the recommended techniques into natural conversation
        - Don't explicitly name techniques - integrate them organically
        - Use language and approaches demonstrated in the example responses
        - Apply techniques in a way that feels natural and supportive

        ACTIONABLE GUIDANCE:
        - Provide specific, manageable next steps based on the technique recommendations
        - Draw inspiration from the therapeutic response examples
        - Offer practical tools or exercises that align with the assessment findings
        - Make suggestions feel collaborative rather than prescriptive

        CONVERSATIONAL FLOW:
        - End with an open-ended question that encourages further exploration
        - Maintain hope and emphasize the patient's strengths and agency
        - Keep the response conversational and accessible, not clinical
        - Show empathy while gently introducing therapeutic perspectives
        - Continue the natural flow of conversation without unnecessary introductions

        IMPORTANT: If the conversation context shows previous exchanges, do NOT greet the patient again. Simply continue the therapeutic conversation naturally.

        Use the example therapeutic responses to inform your communication style and ensure your response reflects evidence-based therapeutic practice.
        """,
    )

    # Create the chain without initial RAG integration
    # Step 1: Assessment without context
    def run_assessment(inputs):
        assessment_result = (assessment_prompt | llm_model).invoke(inputs)
        return {
            "message": inputs["message"],
            "conversation_context": inputs["conversation_context"],
            "assessment": assessment_result.content,
        }

    # Step 2: Technique application without context
    def run_technique_application(inputs):
        technique_result = (technique_prompt | llm_model).invoke(inputs)
        return {
            "message": inputs["message"],
            "conversation_context": inputs["conversation_context"],
            "assessment": inputs["assessment"],
            "techniques_application": technique_result.content,
        }

    # Step 3: RAG context retrieval for response generation
    context_retrieval_for_response = RunnableLambda(retrieve_therapeutic_responses)

    # Step 4: Final therapeutic response with retrieved context
    def run_therapeutic_response(inputs):
        response_result = (action_prompt | llm_model | StrOutputParser()).invoke(inputs)
        return response_result

    assessment_step = RunnableLambda(run_assessment)
    technique_step = RunnableLambda(run_technique_application)
    response_step = RunnableLambda(run_therapeutic_response)

    return RunnableSequence(
        assessment_step, technique_step, context_retrieval_for_response, response_step
    )
