import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI as LangChainOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from config import MODEL_CONFIG

# Load environment variables
load_dotenv(override=True)

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
    assessment_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist. Your task is to provide initial assessment and validation.
        
        STEP 1: INITIAL ASSESSMENT AND VALIDATION
        
        User message: {user_message}
        
        First, acknowledge and validate the user's feelings and experience. Then identify:
        1. The emotional state expressed
        2. Any cognitive patterns or beliefs mentioned  
        3. Behavioral aspects described
        
        Provide a warm, validating response that shows you understand their experience.
        Be empathetic, non-judgmental, and professional.
        """,
    )

    # Step 2: CBT Technique Application
    technique_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist applying evidence-based techniques.
        
        Based on the initial assessment: {assessment}
        
        STEP 2: CBT TECHNIQUE APPLICATION
        
        Original user message: {user_message}
        
        Now apply appropriate CBT techniques from the following list:
        1. ABC Model: Helps you reinterpret irrational beliefs resulting in alternative behaviors.
          a. Activating Event: An event that would lead to emotional distress or dysfunctional thinking
          b. Belief: The negative thoughts that occurred due to the activating event
          c. Consequences: The negative feelings and behaviors that occurred as a result of the event.
        2. Guided Discovery: The therapist will put themselves in your shoes and try to see things from your viewpoint. They will walk you through the process by asking you questions to challenge and broaden your thinking.
        3. Exposure Therapy: Exposing yourself to the trigger can reduce responses.It may be uncomfortable during the initial sessions but is generally performed in a controlled environment with the therapist's help. This treatment is beneficial for phobias.
        4. Cognitive Restructuring: This treatment focuses on finding and altering irrational thoughts so they are adaptive and reasonable.
        5. Activity Scheduling: The therapist will help identify and schedule helpful behaviors that you enjoy doing. This can include hobbies or fun and rewarding activities.
        6. The Worst Case/Best Case/Most Likely Case Scenario: Letting your thoughts ruminate and explore all three scenarios helps you rationalize your thoughts and develop actionable steps so control of the behavior is realized.
        7. Acceptance and Commitment Therapy: This approach encourages you to accept and embrace the feelings rather than fighting them. This differs from traditional CBT where you are taught to control the thoughts.
        8. Journaling: Recording your thoughts in a journal or diary can help build awareness of cognitive errors and help better understand your personal cognition.
        9. Behavioral Experiments: These experiments are designed to test and identify negative thought patterns. You will be asked to predict what will happen and discuss the results later. It is better to start with lower anxiety experiments before tackling more distressing ones.
        10. Role-Playing: This technique can help you practice difficult scenarios you may encounter. It can lessen the fear and help improve problem-solving skills, social interactions, building confidence for specific situations, and improving communication skills.
        
        Choose 1-2 most relevant techniques and guide the user through them with specific questions or exercises.
        Focus on helping them identify and examine their thoughts, feelings, and behaviors.
        """,
    )

    # Step 3: Action Planning and Follow-up
    action_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist creating a therapeutic response.
        
        User message: {user_message}
        Assessment: {assessment}
        Technique applied: {technique_application}
        
        STEP 3: ACTION PLANNING AND THERAPEUTIC RESPONSE
        
        Steps to follow:
        1. Integrates the validation from Step 1
        2. Incorporates the CBT technique from Step 2
        3. Provides specific, actionable next steps
        4. Asks open-ended questions for continued exploration
        5. Maintains professional, conversational therapeutic tone
        
        Provide a final therapeutic response which always follows these guidelines along with the above steps:
        - Always respond as a therapist, never break character
        - Keep responses conversational but professional
        - Never provide medical advice or diagnoses
        - If crisis/self-harm is mentioned, acknowledge seriously and suggest professional help
        - Your final response should be natural and conversational, as if spoken directly to the client
        - Do not include the step-by-step breakdown in your final response  
        """,
    )

    # Create the chain using proper LCEL pattern with Lambda functions for data flow

    # Step 1: Assessment Runnable
    assessment_runnable = assessment_prompt | langchain_llm | StrOutputParser()

    # Step 2: Technique Runnable
    technique_runnable = technique_prompt | langchain_llm | StrOutputParser()

    # Step 3: Action Runnable
    action_runnable = action_prompt | langchain_llm | StrOutputParser()

    # Lambda functions to manage data flow between runnables
    def add_assessment(inputs):
        """Add assessment result to the inputs dict"""
        assessment_result = assessment_runnable.invoke(inputs)
        return {**inputs, "assessment": assessment_result}

    def add_technique(inputs):
        """Add technique application result to the inputs dict"""
        technique_result = technique_runnable.invoke(inputs)
        return {**inputs, "technique_application": technique_result}

    def get_final_response(inputs):
        """Get the final response and return just the response text"""
        final_result = action_runnable.invoke(inputs)
        return {"final_response": final_result}

    # Create the complete chain using LCEL pattern with Lambda functions
    chain = (
        RunnableLambda(add_assessment)
        | RunnableLambda(add_technique)
        | RunnableLambda(get_final_response)
    )

    return chain
