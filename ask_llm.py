from transformers import pipeline

def get_llm_answer(context: str, question: str, model: str = "gpt2"):
    """
    Generate an answer using a Hugging Face model based on the provided context and question.
    
    Args:
        context (str): The context or passage to be used for answering the question.
        question (str): The question that needs to be answered.
        model (str): The Hugging Face model to use (default is 'gpt2').
        
    Returns:
        str: The generated answer from the model.
    """
    
    # Define system and user prompts
    SYSTEM_PROMPT = """
    Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
    """
    
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """
    
    # Initialize the Hugging Face pipeline for text generation
    generator = pipeline("text-generation", model=model)

    # Combine system and user prompts
    input_prompt = SYSTEM_PROMPT + "\n" + USER_PROMPT

    # Generate the response using the model
    response = generator(input_prompt, max_length=200, num_return_sequences=1)

    # Extract the generated answer from the response
    answer = response[0]['generated_text']
    
    print(f"Answer: {answer}")
    return answer

