from groq import Groq
import os
import dotenv

dotenv.load_dotenv()

client = Groq(
    api_key=""
)
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
      {
        "role": "user",
        "content": "what is 1+1?"
      }
    ],
    temperature=1
)

for chunk in completion:
    print(chunk)
