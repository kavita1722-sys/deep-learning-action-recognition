"""
llm_explainer.py

This script is responsible for generating short natural language
explanations for the predicted human actions in my project
"Explainable Human Action Recognition with CNN-LSTM".

After the deep learning model predicts an action (for example:
Basketball, PushUps, TaiChi, etc.), I wanted the system to also
provide a simple explanation describing what that action means.
To achieve this, I integrated a lightweight Large Language Model
using the HuggingFace Transformers library.

Key idea of this script:

1. Text Generation Model
   I use the pretrained "distilgpt2" model because it is smaller,
   faster, and suitable for generating short text explanations.
   The model is loaded using the Transformers pipeline API.

2. Pipeline Initialization
   The pipeline is initialized with:
       pipeline("text-generation", model="distilgpt2")

   I run it on the CPU (device = -1) because explanation generation
   is lightweight and does not require GPU acceleration.

3. Prompt Design
   When an action is predicted, I create a prompt such as:

       "Explain the exercise 'PushUps' in one short sentence."

   This prompt guides the language model to generate a concise
   explanation of the activity.

4. Controlled Text Generation
   I restrict the generation settings to keep the explanation short
   and deterministic:
       • max_new_tokens = 20
       • temperature = 0.3
       • do_sample = False

   This ensures the output remains clear and consistent.

5. Output Processing
   The generated text initially includes the prompt, so I remove
   the prompt portion and return only the explanation.

Purpose of this implementation:

This module helped me understand:
    • How Large Language Models can generate explanations
    • How prompts influence generated text
    • How AI models can be combined (vision + language)
    • How explainability can be added to action recognition systems

This script connects the computer vision model with a language
model to make the system more interpretable and user-friendly.
"""
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="distilgpt2",
    device=-1
)

def explain_action(action):

    prompt = f"Explain the exercise '{action}' in one short sentence."

    result = generator(
        prompt,
        max_new_tokens=20,   # limits generation
        num_return_sequences=1,
        temperature=0.3,
        do_sample=False
    )

    text = result[0]["generated_text"]

    # remove the prompt part
    explanation = text.replace(prompt, "").strip()

    return explanation