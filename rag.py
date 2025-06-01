import os
import signal
import sys
import os
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# create the function that will have the prompt that we send to gemini for context analysis
def generate_rag_prompt(query, context):
    # algorithmic efficiency 
    escaped = context.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""
              You are a helpful and informative bot that answers questions using text from the reference context included below. \
              Be sure to respond in a complete sentence, being comprehensive, including all relavent background information. \
              However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
              strike a friendly and conversational tone. \
              If the context is irrelavent to the answer, you may ignore it.
              QUESTION: '{query}'
              CONTEXT: '{context}'

              ANSWER:
              """).format(query=query, context=context)
    return prompt

# CRUICAL that same embedding function is used for sending to db and querying form db
# k is amt of docs you want to return
# takes the query and performs a similarity search if there is possible 
    # information relating to what we are asking 
def get_db_context(query, k = 6, persist_directory: str = None):

    if persist_directory is None:
        # telling chroma to open existing db from this persist directory
        persist_directory = "./chroma_db_nccn"
    embed_fn = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    db = Chroma(persist_directory=persist_directory, embedding_function=embed_fn)
    results = db.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in results)

# function to send context to gemini
def generate_answer(prompt):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name='gemini-2.0-flash')
    answer = model.generate_content(prompt)
    return answer.text
    
if __name__ == "__main__":
    import signal

    def signal_handler(sig, frame):
        print("thanks for using rag app with gemini")
        sys.exit(0)

    # register Ctrl+C handler (only valid in the main thread)
    signal.signal(signal.SIGINT, signal_handler)

    # simple REPL loop for CLI usage
    while True:
        print("\n" + "-" * 40)
        query = input("What question do you have about the PDF? ")
        ctx = get_db_context(query)
        prompt = generate_rag_prompt(query, ctx)
        ans = generate_answer(prompt)
        print("\n" + ans + "\n")
