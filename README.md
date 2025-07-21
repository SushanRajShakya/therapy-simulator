# Therapy Session Simulator

_AI-Powered CBT Session Simulator with Conversation-Focused RAG_

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-yellow.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📋 Overview

Therapy Session Simulator is an interactive tool designed to replicate realistic conversations based on the principles of **Cognitive Behavioral Therapy (CBT)**. It aims to support mental health professionals, students, and individuals by simulating structured, evidence-based therapeutic sessions.

The simulator enables users to explore CBT techniques such as:

- 🧠 Identifying cognitive distortions
- 💭 Challenging negative thoughts
- 🎯 Practicing behavioral strategies
- 📝 Guided conversational therapeutic interactions

By grounding simulated interactions in established CBT practices and literature, the tool offers a **safe, scalable environment** for training, education, and self-guided reflection.

## 🎯 Objectives

- ✅ **Simulate structured therapy sessions** based on CBT principles
- 🔍 **Use Conversation-focused RAG engine** to ground model responses based on real conversations between users and therapists
- 🔬 **Allow for transparency and debugging** of AI decisions via LangSmith
- 💬 **Provide realistic therapeutic dialogues** for practice or guided self-help
- 📊 **Give a final review of the patient** for further actions or steps to take

## 🚨 Problem Statement

### Current Challenges:

- **Limited Access**: Mental health professionals, students, and individuals seeking self-guided help often lack access to interactive feedback and practice opportunities
- **Adult Stress**: Many adults experience stress from work, family, health, or relationships but don't have established practices for visiting therapists
- **Stigma & Denial**: Fear of admitting psychological problems leads many to stay in denial rather than seek help
- **Inadequate Tools**: Traditional chatbots lack psychological grounding, transparency in reasoning, and fail to incorporate trusted therapeutic frameworks

## 💡 Solution

An **AI-powered Therapy Session Simulator** that leverages **Retrieval-Augmented Generation (RAG)** based on actual conversations between therapists and patients to simulate structured, realistic, and evidence-informed therapy sessions.

### 🏗️ System Architecture

The system features:

- 🦜 **LangChain** for orchestration of agents, memory, and document retrieval
- 📌 **Pinecone** as vector database to store and retrieve CBT-specific knowledge chunks
- 🔍 **LangSmith** for end-to-end observability, debugging, evaluation, and prompt management
- 🎨 **Streamlit** frontend for intuitive and user-friendly interface
- ⚡ **FastAPI** backend for fast and modular API services
- 💬 **Chat-like UI** for natural therapeutic conversations
- 📋 **Session Analysis** with final diagnosis and recommendations

## 🛠️ Tech Stack

| Component          | Technology                    | Purpose                                  |
| ------------------ | ----------------------------- | ---------------------------------------- |
| **Frontend**       | Streamlit                     | User interface and chat experience       |
| **Backend**        | FastAPI                       | API services and business logic          |
| **Orchestration**  | LangChain                     | Agent coordination and memory management |
| **Vector DB**      | Pinecone                      | Knowledge storage and retrieval          |
| **Observability**  | LangSmith                     | Debugging and prompt management          |
| **Language Model** | OpenAI GPT / Anthropic Claude | Conversational AI                        |

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API Key or Anthropic API Key
- Pinecone API Key
- LangSmith API Key (optional, for debugging)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/SushanRajShakya/therapy-simulator.git
   cd therapy-simulator
   ```

2. **Set up virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application**

   ```bash
   # Start FastAPI backend
   cd server && uvicorn main:app --reload

   # Start Streamlit frontend (in new terminal)
   cd frontend && streamlit run main.py
   ```

## 🎮 Usage

1. **Access the Application**: Navigate to `http://localhost:8501`
2. **Start a Session**: Begin a new therapy simulation session
3. **Engage in Conversation**: Chat with the AI therapist using CBT principles
4. **Review Session**: Get analysis and recommendations at session end

## 🔗 References

- [Cognitive Behavioral Therapy (CBT) Principles](https://www.apa.org/ptsd-guideline/patients-and-families/cognitive-behavioral)
- [LangChain Documentation](https://langchain.readthedocs.io/)
- [Pinecone Vector Database](https://www.pinecone.io/)
