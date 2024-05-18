from utils import get_circle_api_key
cci_api_key = get_circle_api_key()
from utils import get_gh_api_key
gh_api_key = get_gh_api_key()
from utils import get_openai_api_key
openai_api_key = get_openai_api_key()

from utils import get_repo_name
course_repo = get_repo_name()
from utils import get_branch
course_branch = get_branch()


from langchain.chat_models import ChatOpenAI
# sampling temperature used to control randomness of model output
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
from prompts import chat_prompt


# parser
from langchain.schema.output_parser import StrOutputParser
output_parser = StrOutputParser()

# langchain expression language allows us to quickly chain functions w/ pipe operator
# https://python.langchain.com/v0.1/docs/expression_language/get_started/
chain = chat_prompt | llm | output_parser

from evals import eval_expected_words
question  = "Generate a quiz about science."
expected_words = ["davinci", "telescope", "physics", "curie"]
expected_words = ["davinci", "telescope", "physics", "curie"]
eval_expected_words(
    prompt_template,
    question,
    expected_words
)

