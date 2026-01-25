# playground.py
from agno.app.playground import Playground
from intent_parser_agent import intent_parser_agent  # ✅ Agent objesini import et

app = Playground(agents=[intent_parser_agent]).get_app()