%%writefile app.py

from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_huggingface import HuggingFacePipeline
from langchain_community.chat_models import ChatAnthropic

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import DocArrayInMemorySearch

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.schema import Document
from langchain import hub

from langchain.embeddings import CohereEmbeddings

from sklearn.mixture import GaussianMixture
import numpy as np
import umap

import os


class KnowledgeRepoBot:
    def __init__(self, model_type='huggingface', model_name='google/gemma-2b', knowledge_file='knowledge.md'):
        self.model_name = model_name
        self.knowledge_file = knowledge_file
        self.model_type = model_type

        if self.model_type == 'huggingface':
            self.model = HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text-generation",
                pipeline_kwargs={
                    "max_new_tokens": 100,
                    "top_k": 50,
                    "temperature": 0.1,
                },
            )
        elif self.model_type == 'claude':
            self.model = ChatAnthropic(anthropic_api_key=self.anthropic_api_key, model="claude-v1")
        else:
            raise ValueError("Unsupported model type. Choose either 'huggingface' or 'claude'.")

    def embed_knowledge(self, embedding_model='google/gemma-2b'):
        # Load in the information
        loader = TextLoader(self.knowledge_file)
        documents = loader.load()
        content = documents[0].page_content

        chunk_size_tok = 1000
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        texts_split = markdown_splitter.split_text(content)

        self.embd = CohereEmbeddings(cohere_api_key=)
        
        #HuggingFaceEmbeddings(
        #    model_name=embedding_model,
        #    multi_process=True,
        #    model_kwargs={"device": "cuda"},
        #    encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        #)

        #results = recursive_embed_cluster_summarize(texts_split, embd, self.model)

        #all_texts = texts_split.copy()

        # Iterate through the results to extract summaries from each level and add them to all_texts
        #for level in sorted(results.keys()):
        #    # Extract summaries from the current level's DataFrame
        #    summaries = results[level][1]["summaries"].tolist()
        #    # Extend all_texts with the summaries from the current level
        #    all_texts.extend(summaries)

        # Now, use all_texts to build the vectorstore with Chroma
        ids = [str(i) for i in range(1, len(texts_split) + 1)]
        try:
            self.vectorstore = Chroma.from_documents(texts_split, embedding=self.embd, ids=ids)
        except Exception:
            Chroma().delete_collection()
            self.vectorstore = Chroma.from_documents(texts_split, embedding=self.embd)
        self.retriever = self.vectorstore.as_retriever()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def get_knowledge(self, question, experience="beginner"):

        prompt = hub.pull("rlm/rag-prompt")

        qa = ConversationalRetrievalChain.from_llm(
            llm=self.model,
            retriever=self.retriever,
            condense_question_prompt=prompt,
            memory=self.memory,
        )
        result = qa({"question": question})

        #response = rag_chain.invoke({"question": question})
        return result["result"]
    def clean_knowledge(self, new_knowledge):
          template = """
              You are the maintainer of an open source knowledge repository which uses markdown.
              Someone wants to add this: {knowledge}.
              Clean it up.
          """
          prompt = PromptTemplate(
              template=template,
              input_variables=['knowledge']
          )
          chain = prompt | self.model

          response = chain.invoke({"knowledge": new_knowledge})
          return response

    def add_knowledge(self, new_knowledge):
        #new_knowledge = self.clean_knowledge(new_knowledge)
        similar_docs = self.vectorstore.similarity_search_with_score(new_knowledge)

        closest_doc = similar_docs[0]
        closest_doc_score = closest_doc[1]
        closest_doc_metadata = closest_doc[0].metadata
        closest_doc = closest_doc[0].page_content

        if closest_doc_score > 1.2:

            updated_content = closest_doc + "\n\n" + new_knowledge
            updated_content = Document(page_content=updated_content)
            self.vectorstore.add_documents([updated_content])
            #for doc in self.vectorstore.get():
                #print(doc)
                #if doc.metadata.get('Header 2') == closest_doc_metadata["Header 2"]:
                    #self.vectorstore.delete(doc.id)
            
        else:
            # If no similar document is found, add the new knowledge as a new document
            self.vectorstore.add_documents([new_knowledge])
        return f"Knowledge Added: {new_knowledge}"

    def get_intro(self, experience):
        """
        Get an introduction to the knowledge repository.
        """

        template = """
            You are the maintainer of an open source knowledge repository.
            Someone with a {experience} level of experience is looking to get up to speed with your work!
            Give a quick overview of the repository.
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=['experience']
        )
        chain = prompt | self.model

        response = chain.invoke({"experience": experience})
      
    def construct_document(self):
        """
        Construct a document from the knowledge repository. 
        """

        model = ChatAnthropic(anthropic_api_key=self.anthropic_api_key, model="claude-v1")

        all_docs = []
        print(self.vectorstore.get())
        for doc in self.vectorstore.get()['documents']:
            all_docs.append(doc)

        all_info = "\n\n".join(all_docs)

        template = """
            You are the maintainer of an open source knowledge repository. The text of which is here: {knowledge}.
            Compile it into a single markdown document which has headers and subsections
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=['knowledge']
        )
        chain = prompt | model

        response = chain.invoke({'knowledge': all_info})
        print(response)
        return response

import streamlit as st
from langchain.llms import OpenAI

st.title('Knowledge Repo')
st.text("If one of the buttons is red, it's probably just working in the background. This isnt' the fastest")
bot = KnowledgeRepoBot()
bot.embed_knowledge()

def generate_response(input_text):
  st.info(bot.get_knowledge(input_text))

def add_knowledge(new_knowledge):
  st.info(bot.add_knowledge(new_knowledge))

def create_document():
  st.write(bot.construct_document())
             

with st.form('my_form'):
  text = st.text_area('Enter text:', 'What are the three key things I need to know about floats?')
  submitted = st.form_submit_button('Submit')
  if submitted:
    generate_response(text) 
    st.balloons()
  
  text2 = st.text_area('Enter new knowledge:', """
      # Exception handling

      This chapter will discuss different types of errors and how to handle some of the them within the program gracefully. You'll also see how to raise exceptions programmatically.

      ## Syntax errors

      Quoting from [docs.python: Errors and Exceptions](https://docs.python.org/3/tutorial/errors.html):

      > There are (at least) two distinguishable kinds of errors: _syntax errors_ and _exceptions_

      Here's an example program with syntax errors:

      ```ruby
      # syntax_error.py
      print('hello')

      def main():
          num = 5
          total = num + 09
          print(total)

      main)
      ```

      The above code is using an unsupported syntax for a numerical value. Note that the syntax check happens before any code is executed, which is why you don't see the output for the `print('hello')` statement. Can you spot the rest of the syntax issues in the above program?

      ```bash
      $ python3.9 syntax_error.py
        File "/home/learnbyexample/Python/programs/syntax_error.py", line 5
          total = num + 09
                        ^
      SyntaxError: leading zeros in decimal integer literals are not permitted;
                  use an 0o prefix for octal integers
      ```
  """)
  submitted2 = st.form_submit_button('Add Knowledge')
  if submitted2:
    add_knowledge(text2)
  
  submitted3 = st.form_submit_button('Create Document')
  if submitted3:
    create_document()
    
