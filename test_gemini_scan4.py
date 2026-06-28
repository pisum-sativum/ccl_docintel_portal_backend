import os
import sys
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

load_dotenv(r'c:\Users\KIIT0001\Desktop\CCL-Project\ccl-docintel-backend\.env')

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0.2,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

compliance_prompt = (
    f"You are an industrial compliance auditor. Analyze the following document text "
    f"and its filename ('hack.pdf') for any operational hazards, safety violations, "
    f"missing safety protocols, financial discrepancies, expired certifications, "
    f"or any errors, inconsistencies, or suspicious patterns in the filename itself.\n\n"
    f"Respond in exactly this format:\n"
    f"RISK: [High, Medium, or None]\n"
    f"REASON: [A short 1-sentence description of the hazard found]\n\n"
    f"--- DOCUMENT TEXT ---\n{'hack malicious'}"
)

try:
    response = llm.invoke(compliance_prompt)
    print(response.content)
except Exception as e:
    print('EXCEPTION:', repr(e))
