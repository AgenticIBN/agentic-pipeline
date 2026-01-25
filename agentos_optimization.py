# agentos_optimization.py
from dotenv import load_dotenv
load_dotenv()

from agno.os import AgentOS
from optimization_agent import optimization_agent # senin optimization agent objen

agent_os = AgentOS(agents=[optimization_agent])
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="agentos_optimization:app", reload=True, port=8001)