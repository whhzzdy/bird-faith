import json
import os
import time
import threading
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= CONFIGURATION =================
API_KEY = "sk-9885898f801645a890417229eeb56d78"
BASE_URL = "https://api.deepseek.com"

INPUT_FILE = "bird_train/all/cleaned_predict_dev_with_cot.json"
OUTPUT_FILE = "decompose/decomposed_cot.json"

MAX_WORKERS = 10
MAX_RETRIES = 3
# =================================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
result_lock = threading.Lock()

def decompose_cot(cot_text):
    system_prompt = """You are a highly precise Text Segmentation Assistant.
Your task is to take a monolithic Chain of Thought (CoT) and decompose it into a JSON array of independent, self-contained logical steps.

**CRITICAL RULES:**
1. **Handling Turning Points & Self-Corrections (CRUCIAL)**: 
   - NEVER split sentences or clauses connected by contrasting words like "but", "however", "wait", "actually", or "except". 
   - A condition and its caveat/correction represent a SINGLE logical action. 
   - You MUST restructure or combine these clauses into one cohesive sentence. 
   - *Example Input*: "I should join table A and B, but actually table B doesn't have the column, so I will join table C."
   - *Correct Step Output*: "I initially considered joining table A and B, but recognized that table B lacks the column, so I decided to join table A and C instead."

2. **Granularity**: Each step should represent one complete logical milestone (e.g., identifying a target table, formulating a filter condition, or planning a specific join). Do not over-fragment ideas that belong together.

3. **Self-Containment (Pronoun Resolution)**: If a sentence uses pronouns (e.g., "Join them", "Filter this"), replace them with the actual tables/columns mentioned previously so the step makes sense completely on its own. 

4. **Faithfulness**: Do not add external knowledge or hallucinate. Preserve the exact analytical intent, but you are ALLOWED to adjust grammar and sentence structure to satisfy Rules 1 and 3.

**OUTPUT FORMAT:**
Return ONLY a valid JSON object strictly matching this schema:
{
    "steps":[
        {"step_id": 1, "content": "First cohesive logical step..."},
        {"step_id": 2, "content": "Second cohesive logical step..."}
    ]
}"""
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Decompose the following CoT:\n\n{cot_text}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            raw_out = response.choices[0].message.content
            start = raw_out.find('{')
            end = raw_out.rfind('}') + 1
            return json.loads(raw_out[start:end]).get("steps",[])
            
        except Exception as e:
            time.sleep(2)
            
    return[{"step_id": 1, "content": "API/Parsing Error during decomposition."}]

def process_item(item):
    q_id = item.get("question_id")
    original_cot = item.get("thought_process", "")
    
    if not original_cot:
        return q_id, item,[]
        
    steps = decompose_cot(original_cot)
    return q_id, item, steps

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return
        
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    results =[]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(process_item, item): item for item in data}
        
        for future in tqdm(as_completed(future_to_item), total=len(data), desc="Decomposing"):
            q_id, original_item, decomposed_steps = future.result()
            
            new_item = original_item.copy()
            new_item["decomposed_steps"] = decomposed_steps
            
            with result_lock:
                results.append(new_item)
                 
    results.sort(key=lambda x: int(x.get("question_id", 0)) if str(x.get("question_id", 0)).isdigit() else str(x.get("question_id")))
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"Saved decomposed output to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()