import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser


from server.config import MODEL_CONFIG

# Load environment variables
load_dotenv(override=True)

# Configure OpenAI
open_ai_api_key = os.getenv("OPENAI_API_KEY")

# Configure LangChain LLM
llm_model = ChatOpenAI(
    api_key=open_ai_api_key,
    model=MODEL_CONFIG["model"],
    temperature=MODEL_CONFIG["temperature"],
)


# Create CBT Sequential Chain
def create_cbt_sequential_chain():
    # Step 1: Initial Assessment and Validation
    assessment_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist. Your task is to provide initial assessment and sentiment analysis of the patient.

        Patient message: {message}

        Identify:
        1. The emotional state expressed
        2. Any cognitive patterns or beliefs mentioned
        3. Behavioral aspects described

        Provide key points summarizing the patient's emotional state, cognitive patterns, and behaviors.
        """,
    )

    # Step 2: CBT Technique Application
    technique_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist applying evidence-based techniques to help the patient whose emotional state is described in the assessment enclosed by ###.

        Assessment: ###{assessment}###

        Choose 2 or more relevant CBT techniques and suggest which techniques to focus on based on their assessment.
        
        Techniques to consider are provided below:
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
        """,
    )

    # Step 3: Action Planning and Follow-up
    action_prompt = ChatPromptTemplate.from_template(
        """
        You are a professional CBT therapist creating a therapeutic response based on CBT techniques suggested.

        CBT Techniques suggested based on assessment: {techniques_application}

        Provide a final therapeutic response which always follows these guidelines along with the above steps:
        - Always respond as a therapist, never break character
        - Always maintain a professional, conversational therapeutic tone
        - Never provide medical advice or diagnoses
        - If crisis/self-harm is mentioned, acknowledge seriously and suggest professional help
        - Your final response should be natural and conversational, as if spoken directly to a patient as client
        - Do not include step-by-step breakdown in your final response
        - Provides specific, actionable next steps 
        - Be very kind, supportive, and empathetic in your final response
        - Do not directly use the CBT technique names in your final response, instead integrate them naturally into the conversation
        - Always end with an open-ended question to encourage further discussion
        """,
    )

    assessment = assessment_prompt | llm_model

    apply_cbt_technique = technique_prompt | llm_model

    therapeutic_response = action_prompt | llm_model | StrOutputParser()

    return RunnableSequence(assessment, apply_cbt_technique, therapeutic_response)
