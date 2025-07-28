from typing import Dict, List, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from server.config import *
from server.chat_model import ChatMessage


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=MODEL_CONFIG["model"],
            temperature=0.1,  # Lower temperature for more consistent summaries
        )

    def get_session(self, session_id: str) -> Dict:
        """Get or create a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "messages": [],
                "summary": "",
                "created_at": datetime.now(),
                "last_updated": datetime.now(),
                "message_count": 0,
            }
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the session"""
        session = self.get_session(session_id)
        session["messages"].append(ChatMessage(role=role, content=content))
        session["last_updated"] = datetime.now()
        session["message_count"] += 1

        # Update summary every 6 messages to keep context manageable
        if session["message_count"] % 6 == 0:
            session["summary"] = self._generate_summary(session_id)

    def get_conversation_context(self, session_id: str) -> str:
        """Get conversation context for the LLM - either summary + recent messages or all messages if few"""
        session = self.get_session(session_id)
        messages = session["messages"]

        if len(messages) <= 6:
            # If conversation is short, use all messages
            context = "Full conversation history:\n"
            for msg in messages:
                context += f"{msg.role.title()}: {msg.content}\n"
        else:
            # Use summary + last 4 messages for cost efficiency
            context = f"Conversation Summary: {session['summary']}\n\n"
            context += "Recent conversation:\n"
            for msg in messages[-4:]:
                context += f"{msg.role.title()}: {msg.content}\n"

        # Add conversation state information
        if len(messages) == 0:
            context += "\nCONVERSATION STATE: This is the very first interaction with this patient. A greeting is appropriate."
        elif len(messages) >= 2:
            context += f"\nCONVERSATION STATE: This is an ongoing conversation with {len(messages)//2} previous exchanges. The therapeutic relationship is already established. Do NOT greet the patient again."

        return context

    def classify_message(self, message: str, session_id: str) -> str:
        """Classify user message to determine response strategy"""
        conversation_context = self.get_conversation_context(session_id)

        classification_prompt = ChatPromptTemplate.from_template(
            """
            You are a CBT therapist assistant that classifies patient messages to optimize response strategy.

            Patient message: "{message}"
            
            Conversation context:
            {context}

            Classify this message into ONE of these categories:

            GREETING: Simple hellos, introductions, pleasantries (e.g., "Hi doctor", "Nice to meet you", "How are you?")
            
            PROCEDURAL: Questions about the session process (e.g., "Should we start?", "How does this work?", "What do we do now?")
            
            SESSION_END: Patient wants to end the session (e.g., "Have a good day doc", "See you soon", "I think I feel ok now", "Ready to end", "That's all for today", "I should go", "Thanks for today")
            
            THERAPEUTIC: Meaningful emotional/psychological content that requires full CBT analysis (e.g., sharing feelings, problems, thoughts, experiences, concerns, asking for help with specific issues)

            SMALL_TALK: Casual conversation not requiring therapeutic intervention (e.g., comments about weather, general life updates without emotional content)

            Respond with ONLY the category name: GREETING, PROCEDURAL, SESSION_END, THERAPEUTIC, or SMALL_TALK
            """
        )

        try:
            classification_result = (classification_prompt | self.llm).invoke(
                {"message": message, "context": conversation_context}
            )
            classification = classification_result.content.strip().upper()

            # Validate classification
            valid_classifications = [
                "GREETING",
                "PROCEDURAL",
                "SESSION_END",
                "THERAPEUTIC",
                "SMALL_TALK",
            ]
            if classification in valid_classifications:
                return classification
            else:
                # Default to THERAPEUTIC if classification is unclear
                return "THERAPEUTIC"

        except Exception as e:
            print(f"Error classifying message: {e}")
            # Default to THERAPEUTIC to be safe
            return "THERAPEUTIC"

    def generate_simple_response(
        self, message: str, session_id: str, response_type: str
    ) -> str:
        """Generate simple responses for non-therapeutic messages"""
        conversation_context = self.get_conversation_context(session_id)

        if response_type == "GREETING":
            prompt_template = """
            You are a warm, professional CBT therapist responding to a patient's greeting.
            
            Patient message: "{message}"
            Conversation context: {context}
            
            Provide a brief, warm greeting response that:
            - Acknowledges their greeting warmly
            - Maintains professional therapeutic boundaries
            - Gently transitions toward therapeutic conversation
            - Is contextually appropriate (don't re-introduce yourself if already met)
            
            Keep it brief and natural (1-2 sentences).
            """

        elif response_type == "PROCEDURAL":
            prompt_template = """
            You are a CBT therapist responding to a patient's procedural question.
            
            Patient message: "{message}"
            Conversation context: {context}
            
            Provide a brief, helpful response that:
            - Answers their procedural question
            - Reassures them about the process
            - Encourages them to share what's on their mind
            - Maintains a supportive, professional tone
            
            Keep it brief and encouraging (1-3 sentences).
            """

        elif response_type == "SMALL_TALK":
            prompt_template = """
            You are a CBT therapist responding to casual conversation.
            
            Patient message: "{message}"
            Conversation context: {context}
            
            Provide a brief response that:
            - Acknowledges their comment politely
            - Gently redirects toward therapeutic topics
            - Shows interest in their wellbeing
            - Maintains professional therapeutic focus
            
            Keep it brief and gently redirecting (1-2 sentences).
            """

        simple_prompt = ChatPromptTemplate.from_template(prompt_template)

        try:
            response_result = (simple_prompt | self.llm).invoke(
                {"message": message, "context": conversation_context}
            )
            return response_result.content
        except Exception as e:
            print(f"Error generating simple response: {e}")
            return (
                "Thank you for sharing that. What would you like to talk about today?"
            )

    def _generate_summary(self, session_id: str) -> str:
        """Generate a therapeutic summary of the conversation"""
        session = self.get_session(session_id)
        messages = session["messages"]

        # Create conversation text
        conversation_text = ""
        for msg in messages:
            conversation_text += f"{msg.role.title()}: {msg.content}\n"

        summary_prompt = ChatPromptTemplate.from_template(
            """
            You are a professional CBT therapist creating a therapeutic summary of a conversation.
            
            Conversation to summarize:
            {conversation}
            
            Create a concise therapeutic summary covering:
            1. KEY CONCERNS: Main issues the patient has discussed
            2. EMOTIONAL PATTERNS: Primary emotions and mood patterns observed
            3. COGNITIVE PATTERNS: Thought patterns, beliefs, and cognitive distortions identified
            4. BEHAVIORAL PATTERNS: Behaviors, coping mechanisms, and avoidance patterns
            5. THERAPEUTIC PROGRESS: CBT techniques applied and patient's responses
            6. IMPORTANT CONTEXT: Key background information and triggers mentioned
            
            Keep the summary clinical but empathetic, focusing on information that would help continue effective therapy.
            Maximum 300 words.
            """
        )

        try:
            summary_result = (summary_prompt | self.llm).invoke(
                {"conversation": conversation_text}
            )
            return summary_result.content
        except Exception as e:
            print(f"Error generating summary: {e}")
            # Fallback to basic summary
            return f"Patient has discussed various concerns over {len(messages)} messages. Key themes include emotional and behavioral challenges that require continued therapeutic support."

    def generate_session_conclusion(self, session_id: str) -> str:
        """Generate a final conclusion/diagnosis for the session"""
        session = self.get_session(session_id)
        context = self.get_conversation_context(session_id)

        conclusion_prompt = ChatPromptTemplate.from_template(
            """
            You are a professional CBT therapist providing a final session summary and therapeutic conclusion.
            
            Session context:
            {context}
            
            Provide a warm, professional session conclusion that includes:
            
            1. ACKNOWLEDGMENT: Recognize the patient's openness and courage in sharing
            2. KEY INSIGHTS: Summarize the main patterns and insights discovered
            3. PROGRESS NOTED: Highlight any positive steps or awareness gained
            4. THERAPEUTIC RECOMMENDATIONS: Suggest continued focus areas (without being prescriptive)
            5. ENCOURAGEMENT: Offer hope and validation for their therapeutic journey
            
            Guidelines:
            - Be warm, supportive, and professional
            - Avoid clinical jargon - use accessible language
            - Do NOT provide medical diagnoses
            - Do NOT ask questions - this is a conclusion
            - End with encouragement about their therapeutic journey
            - Keep it concise but meaningful (200-300 words)
            
            This is the final message of the session, so provide closure and hope.
            """
        )

        try:
            conclusion_result = (conclusion_prompt | self.llm).invoke(
                {"context": context}
            )
            return conclusion_result.content
        except Exception as e:
            print(f"Error generating conclusion: {e}")
            return "Thank you for sharing so openly today. Your willingness to explore your thoughts and feelings shows real courage. Continue to be patient and kind with yourself as you work through these challenges. Remember that growth takes time, and you're taking important steps forward."

    def clear_session(self, session_id: str):
        """Clear a session (optional - for cleanup)"""
        if session_id in self.sessions:
            del self.sessions[session_id]


# Global session manager instance
session_manager = SessionManager()
