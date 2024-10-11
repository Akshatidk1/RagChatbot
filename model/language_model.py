import openai

def generate_response(query, documents):
    openai.api_key = "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"
    
    # Combine query with relevant documents
    context = "\n".join(documents)
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=150)
    return response.choices[0].text.strip()
