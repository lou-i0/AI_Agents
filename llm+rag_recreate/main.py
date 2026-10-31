#=====================================
# %% Import Relevant Libraries
#=====================================
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever           # Import from the vector.py file
#=====================================
# %% Specify the model to use
#=====================================
model = OllamaLLM(model="llama3.2")

#=====================================
# %% invoke the model with context on what it is , and what it should do
#=====================================
template = """ Hey there Llama! You are a eccentric and witty assistant (think like alfred from batman) named D.A.V.E (Data Analysis Virtual Executive)

that is here to help me with my questions. 

Additionally, you are a master of puns and wordplay with some sass :), so feel free to sprinkle in some humor when appropriate.
You will have access to some documentation to review which will aid you in answering my questions. Please provide your responses in a clear and concise manner, and don't hesitate to ask for clarification if needed. Let's have some fun while we work together!

Here is some Relevant documentation you may want to go through before answering my question : {documentation}

Here is my question for you good sir! : {question}

Base your answer on the retrieved documentation. If it does not contain the answer, say so rather than guessing. Cite the source and page when available.

Whilst you may have a lovely personality, you act in this way:

1) keep things short and punchy when needed
2) be honest and transparent about what you know and don't know (' A Harsh truth is better than a pretty lie')
3) otherwise , have fun!

"""
#=====================================
# %% invoke the prompt template with the model
#=====================================
prompt = ChatPromptTemplate.from_template(template = template)

#=====================================
# %% Create a chain 
#=====================================
chain = prompt | model

#=====================================
# %% Compute and show the result - with while loop to keep asking questions until user types 'quit'
#=====================================
while True:
    print('='*50)
    question = input(" Hello good Fellow! What is your question for me today? (Type q to 'quit'): ")
    print('='*50)
    print('\n\n')
    if question.lower() == 'q':
        print("I trust you now have all the answers you need, farewell!")
        break

    retrieved_documents = retriever.invoke(question)
    documentation_parts = []
    for document in retrieved_documents:
        metadata = document.metadata
        location = f"Source: {metadata.get('source', 'unknown')}"
        if isinstance(metadata.get('page'), int):
            location += f", page {metadata['page'] + 1}"
        elif metadata.get('row') is not None:
            location += f", row {metadata['row']}"
        if isinstance(metadata.get('chunk_index'), int):
            location += (
                f", chunk {metadata['chunk_index'] + 1}/{metadata.get('chunk_count', '?')}"
                f", {metadata.get('chunk_chars', len(document.page_content))} characters"
            )
        documentation_parts.append(f"{location}\n{document.page_content}")
    documentation = "\n\n".join(documentation_parts)
    if not documentation:
        documentation = "No relevant documentation was retrieved."

    result = chain.invoke({"question": question, "documentation": documentation})
    print(result)
