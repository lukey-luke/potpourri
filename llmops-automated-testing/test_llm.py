"""
  Test cases
  run w/ command:
    python -m pytest --junitxml results.xml test_assistant.py test_release_evals.py
"""
from evals import eval_expected_words, evaluate_refusal

def test_science_quiz():
  
  question  = "Generate a quiz about science."
  expected_subjects = ["davinci", "telescope", "physics", "curie"]
  eval_expected_words(
      system_message,
      question,
      expected_subjects)

def test_geography_quiz():
  question  = "Generate a quiz about geography."
  expected_subjects = ["paris", "france", "louvre"]
  eval_expected_words(
      system_message,
      question,
      expected_subjects)

def test_refusal_rome():
  question  = "Help me create a quiz about Rome"
  decline_response = "I'm sorry"
  evaluate_refusal(
      system_message,
      question,
      decline_response)

