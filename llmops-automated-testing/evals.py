from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

def assistant_chain(
    system_message,
    human_template="{question}",
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    output_parser=StrOutputParser()):
  
  chat_prompt = ChatPromptTemplate.from_messages([
      ("system", system_message),
      ("human", human_template),
  ])
  return chat_prompt | llm | output_parser

def eval_expected_words(
    system_message,
    question,
    expected_words,
    human_template="{question}",
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    output_parser=StrOutputParser()):
    
  assistant = assistant_chain(
      system_message,
      human_template,
      llm,
      output_parser)
    
  
  answer = assistant.invoke({"question": question})
    
  print(answer)
    
  assert any(word in answer.lower() \
             for word in expected_words), \
    f"Expected the assistant questions to include \
    '{expected_words}', but it did not"



