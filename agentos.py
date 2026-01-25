# agentos.py
from dotenv import load_dotenv
load_dotenv()  # .env'den GROQ_API_KEY okunsun

from agno.os import AgentOS
from intent_parser_agent import intent_parser_agent

agent_os = AgentOS(agents=[intent_parser_agent])
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="agentos:app", reload=True)