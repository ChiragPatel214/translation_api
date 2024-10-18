import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from flores200_codes import flores_codes

app = FastAPI()

# Dictionary to store models and tokenizers
model_dict = {}

# Use the older @app.on_event("startup") and @app.on_event("shutdown") approach
@app.on_event("startup")
async def startup_event():
    # Load models and tokenizers at startup
    print("Loading models...")
    model_name_dict = {'nllb-distilled-600M': 'facebook/nllb-200-distilled-600M'}
    
    for call_name, real_name in model_name_dict.items():
        model = AutoModelForSeq2SeqLM.from_pretrained(real_name)
        tokenizer = AutoTokenizer.from_pretrained(real_name)
        model_dict[call_name + '_model'] = model
        model_dict[call_name + '_tokenizer'] = tokenizer
        print(f"Loaded model: {call_name}")

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down the app and cleaning up resources...")

# Define a Pydantic model for the request body
class TranslationRequest(BaseModel):
    source: str  # or source_lang if you chose Option 1
    target: str  # or target_lang if you chose Option 1
    text: str


@app.post("/translate")
async def translate(request: TranslationRequest):
    print("Received request:", request.dict())  # Check the received request
    # Check if source and target languages are valid
    if request.source not in flores_codes or request.target not in flores_codes:
        raise HTTPException(status_code=400, detail="Invalid source or target language")

    source_code = flores_codes[request.source]
    target_code = flores_codes[request.target]

    model_name = 'nllb-distilled-600M'
    model = model_dict[model_name + '_model']
    tokenizer = model_dict[model_name + '_tokenizer']

    translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=source_code, tgt_lang=target_code)

    start_time = time.time()
    output = translator(request.text, max_length=400)
    translated_text = output[0]['translation_text']
    end_time = time.time()

    return {
        "inference_time": end_time - start_time,
        "source": request.source,
        "target": request.target,
        "result": translated_text
    }

# To run the server: uvicorn app:app --reload
# python -m uvicorn app:app --reload