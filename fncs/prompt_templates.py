""" User/system Prompt Templates"""

def user_prompt():
    user_prompt_template = \
    """
    ***Question: {}
    
    ***Context:
    <--Start of Context-->
    {}
    <--End of Context-->
    
    **Instructions:
    - Answer based ONLY on the provided context above
    - Do not include external knowledge
    - Be concise and specific
    
    **Required Format:
    1. Answer:
       [Your detailed response here]
    
    2. Sources:
       • [Unique Source URL 1]
       • [Unique Source URL 2]
       • [...]
    
    Note: If the answer cannot be determined from the provided context,
    state: "Cannot be determined from the given context."
    """
    return user_prompt_template

def user_prompt_without_context():
    user_prompt_template = \
    """
    ***Question: {}
    
    **Instructions:
    - Be concise and specific
    
    **Required Format:
    1. Answer:
       [Your detailed response here]
    
    2. Sources:
       • [Unique Source URL 1]
       • [Unique Source URL 2]
       • [...]
    
    If the answer cannot be determined, state:
    "Cannot be determined as I do not have enough the knowledge to answer this question."
    """
    return user_prompt_template