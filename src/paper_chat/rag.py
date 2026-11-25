from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from paper_chat.ingest import get_retriever

load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-mini")


retriever = get_retriever(k=3)

# Step 5: Generate Prompt + model

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ("You are a helpful tutor. Answer the question using only the context below."
            "If the answer isn't in the context, say \"I don't know.\"\n\n"
            "Context:\n{context}"),
        ),
        ("human", "Question: {question}"),
    ]
)


def format_docs(docs):
    """Join the retrieved chunks into one block of the text for the prompt"""
    return "\n\n".join([doc.page_content for doc in docs])

# === RAG chain ===
# The question goes two ways:
#   "context":  retriever finds chunks -> format_docs turns them into text
#   "question": RunnablePassthrough passes the question through unchanged

rag_chain = (
    {"context": retriever | format_docs , "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

for question in [
   "Which popular LLMs were considered in the paper?",
    "Which model was the most accurate?",
    "What is the capital of France?",  # not in the documents -> should say "I don't know." 
]: 
    print(f"\nQ: {question}")
    print(f"\nA: {rag_chain.invoke(question)}")