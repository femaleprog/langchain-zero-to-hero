import os
# os.environ["ACTIVELOOP_TOKEN"] = "
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# load the existing Deep Lake dataset and specify the embedding function
db = DeepLake(dataset_path="https:/app.activeloop.ai/activeloop/wiki-art", embedding_function=embeddings)

# create new documents
texts = [
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# add documents to our Deep Lake dataset
db.add_documents(docs)
# instantiate the wrapper class for GPT3
llm = OpenAI(model="text-davinci-003", temperature=0)

# create a retriever from the db
retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm, chain_type="stuff", retriever=db.as_retriever()
)

# instantiate a tool that uses the retriever
tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

# create an agent that uses the tool
agent = initialize_agent(
	tools,
	llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True
)
response = agent.run("When was Michael Jordan born?")
print(response)