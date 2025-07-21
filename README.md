# Therapy Session Simulator

Therapy Session Simulator is an interactive tool designed to replicate realistic conversations based on the principles of **Cognitive Behavioral Therapy (CBT)**. It aims to support mental health professionals, students, and individuals by simulating structured, evidence-based therapeutic sessions.

The simulator enables users to explore CBT techniques such as:

- 🧠 Identifying cognitive distortions
- 💭 Challenging negative thoughts
- 🎯 Practicing behavioral strategies
- 📝 Guided conversational therapeutic interactions

By grounding simulated interactions in established CBT practices and literature, the tool offers a **safe, scalable environment** for training, education, and self-guided reflection.

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

- OpenAI API Key
- Pinecone API Key
- LangSmith API Key

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/SushanRajShakya/therapy-simulator.git
   cd therapy-simulator
   ```

2. **Set up virtual environment**

   Firstly install [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)

   ```bash
   # Create necessary virtual env named 'therapy-simulator'
   conda create -n therapy-simulator python=3.11.5

   # Activate virtual env
   conda activate therapy-simulator
   ```

3. **Install dependencies**

   ```bash
   # Installs necessary dependencies for the project in virtual env
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   ```bash
   # Copy environments from .env.example and edit .env with necessary values
   cp .env.example .env
   ```

5. **Run the application**

   ```bash
   # Start FastAPI backend
   fastapi dev server/main.py

   # Start Streamlit frontend (in new terminal)
   streamlit run ./frontend/main.py
   ```
