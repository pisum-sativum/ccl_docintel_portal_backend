import os
import sys
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv(r'c:\Users\KIIT0001\Desktop\CCL-Project\ccl-docintel-backend\.env')

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY", ""),
    temperature=0.2,
)

def test_scan(extracted_text, filename):
    compliance_prompt = (
        f"You are an industrial compliance auditor. Analyze the following document text "
        f"and its filename ('{filename}') for any operational hazards, safety violations, "
        f"missing safety protocols, financial discrepancies, expired certifications, "
        f"or any errors, inconsistencies, or suspicious patterns in the filename itself.\n\n"
        f"Respond in exactly this format:\n"
        f"RISK: [High, Medium, or None]\n"
        f"REASON: [A short 1-sentence description of the hazard found]\n\n"
        f"--- DOCUMENT TEXT ---\n{extracted_text[:4000]}"
    )
    response = llm.invoke(compliance_prompt)
    print('Response:', response.content)

try:
    test_scan('hack malicious', 'hack.pdf')
except Exception as e:
    traceback.print_exc()
