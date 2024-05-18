# Want to use an LLM assistant to determine if the outputs of another model follow the prompt
from utils import get_circle_api_key, get_gh_api_key, get_openai_api_key
cci_api_key = get_circle_api_key()
gh_api_key = get_gh_api_key()
openai_api_key = get_openai_api_key()

from utils import get_repo_name, get_branch
course_repo = get_repo_name()
course_branch = get_branch()

delimiter = "####"
# system messages are instructions on the broad task for what you're using the model for
eval_system_prompt = f"""You are an assistant that evaluates \
  whether or not an assistant is producing valid quizzes.
  The assistant should be producing output in the \
  format of Question N:{delimiter} <question N>?"""

# This is a mock response before we generate one from an actual LLM
llm_response = """
Question 1:#### What is the largest telescope in space called and what material is its mirror made of?

Question 2:#### True or False: Water slows down the speed of light.

Question 3:#### What did Marie and Pierre Curie discover in Paris?
"""

# user messages are specific examples for the broad task in the system message
eval_user_message = f"""You are evaluating a generated quiz \
based on the context that the assistant uses to create the quiz.
  Here is the data:
    [BEGIN DATA]
    ************
    [Response]: {llm_response}
    ************
    [END DATA]

Read the response carefully and determine if it looks like \
a quiz or test. Do not evaluate if the information is correct
only evaluate if the data is in the expected format.

Output Y if the response is a quiz, \
output N if the response does not look like a quiz.
"""

from langchain.prompts import ChatPromptTemplate
eval_prompt = ChatPromptTemplate.from_messages([
      ("system", eval_system_prompt),
      ("human", eval_user_message),
  ])
from langchain.chat_models import ChatOpenAI
# this llm is for evaluating 
eval_llm = ChatOpenAI(model="gpt-3.5-turbo",
                 temperature=0)
from langchain.schema.output_parser import StrOutputParser
output_parser = StrOutputParser()


# putting it all together
eval_chain = eval_prompt | eval_llm | output_parser
# invoke the chain and store the result in a variable
# should be defined as either 'Y' or 'N'
response = eval_chain.invoke({})


def create_eval_chain(
    agent_response,
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    output_parser=StrOutputParser()
):
  delimiter = "####"
  eval_system_prompt = f"""You are an assistant that evaluates whether or not an assistant is producing valid quizzes.
  The assistant should be producing output in the format of Question N:{delimiter} <question N>?"""
  
  eval_user_message = f"""You are evaluating a generated quiz based on the context that the assistant uses to create the quiz.
  Here is the data:
    [BEGIN DATA]
    ************
    [Response]: {agent_response}
    ************
    [END DATA]

Read the response carefully and determine if it looks like a quiz or test. Do not evaluate if the information is correct
only evaluate if the data is in the expected format.

Output Y if the response is a quiz, output N if the response does not look like a quiz.
"""
  eval_prompt = ChatPromptTemplate.from_messages([
      ("system", eval_system_prompt),
      ("human", eval_user_message),
  ])

  return eval_prompt | llm | output_parser

known_bad_result = "There are lots of interesting facts. Tell me more about what you'd like to know"

bad_eval_chain = create_eval_chain(known_bad_result)
bad_eval_response = bad_eval_chain.invoke({})  # should be 'N'

