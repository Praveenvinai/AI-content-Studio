import os

from dotenv import load_dotenv
from groq import Groq


# ============================================
# LOAD .env
# ============================================

load_dotenv()


# ============================================
# GET API KEY
# ============================================

groq_api_key = os.getenv("GROQ_API_KEY")


# ============================================
# CHECK API KEY
# ============================================

if not groq_api_key:

    print("❌ GROQ_API_KEY was not found.")

    exit()


# ============================================
# CREATE GROQ CLIENT
# ============================================

client = Groq(
    api_key=groq_api_key
)


# ============================================
# SEND FIRST AI REQUEST
# ============================================

response = client.chat.completions.create(

    model="openai/gpt-oss-120b",

    messages=[

        {
            "role": "user",

            "content":
                "Explain artificial intelligence in one simple sentence."
        }

    ],

    temperature=0.7,

    max_tokens=100
)


# ============================================
# GET AI RESPONSE
# ============================================

answer = response.choices[0].message.content


# ============================================
# DISPLAY RESPONSE
# ============================================

print()
print("🤖 Groq AI Response:")
print(answer)