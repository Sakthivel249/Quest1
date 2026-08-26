import json
import re

transcript_path = r"C:\Users\sakth\.gemini\antigravity-ide\brain\c5b640b2-d6ff-4d95-913c-2aa781042717\.system_generated\logs\transcript_full.jsonl"

with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get("type") == "USER_INPUT":
                content = data.get("content", "")
                
                # Extract text within <USER_REQUEST> tags
                match = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", content, re.DOTALL)
                if match:
                    prompt = match.group(1).strip()
                    if prompt:
                        print("---")
                        print(prompt)
        except Exception as e:
            pass
