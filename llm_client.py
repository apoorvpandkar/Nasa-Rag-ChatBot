from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-4o-mini") -> str:
    """Generate response using OpenAI with context"""

    system_prompt = """You are an expert NASA space mission assistant with deep knowledge of 
space exploration history, astronaut missions, spacecraft systems, and mission operations.

You have access to detailed documentation about NASA missions including Apollo 11, Apollo 13, 
the Space Shuttle Challenger, and other historical missions. Use the provided context to give 
accurate, informative answers. When referencing specific facts, mention the source document 
when available.

If the context doesn't contain enough information to fully answer a question, acknowledge 
what you know from the documents and clearly indicate when you're supplementing with 
general knowledge. Always prioritize accuracy and clarity."""

    messages = [{"role": "system", "content": system_prompt}]

    if context:
        messages.append({
            "role": "user",
            "content": f"Here is relevant context from NASA mission documents:\n\n{context}"
        })
        messages.append({
            "role": "assistant",
            "content": "Thank you. I'll use this context to help answer your questions accurately."
        })

    for entry in conversation_history:
        if entry.get("role") in ("user", "assistant"):
            messages.append({"role": entry["role"], "content": entry["content"]})

    messages.append({"role": "user", "content": user_message})

    import os
    client = OpenAI(
        api_key=openai_key,
        base_url=os.environ.get("OPENAI_API_BASE")
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content