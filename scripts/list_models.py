"""Print the LLM models your API keys can actually use.
Model IDs change often — run this if you get a 'model not found' error."""
import os
from dotenv import load_dotenv
load_dotenv()

gk = os.environ.get("GROQ_API_KEY")
if gk:
    try:
        from openai import OpenAI
        c = OpenAI(api_key=gk, base_url="https://api.groq.com/openai/v1")
        print("Groq models available to your key:")
        for m in c.models.list().data:
            print("   ", m.id)
    except Exception as e:
        print("Groq list failed:", e)
else:
    print("No GROQ_API_KEY set — skipping Groq.")
