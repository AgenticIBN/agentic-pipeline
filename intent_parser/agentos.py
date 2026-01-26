# agentos.py
from dotenv import load_dotenv
load_dotenv()  # .env'den GOOGLE_API_KEY okunsun

import sys
import importlib.util

# Load intent_parser_agent.py (underscores in filename)
spec = importlib.util.spec_from_file_location("intent_parser_agent", "intent_parser_agent.py")
module = importlib.util.module_from_spec(spec)
sys.modules["intent_parser_agent"] = module
spec.loader.exec_module(module)

intent_parser_agent = module.intent_parser_agent

from agno.os import AgentOS

agent_os = AgentOS(agents=[intent_parser_agent])
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="agentos:app", reload=True)