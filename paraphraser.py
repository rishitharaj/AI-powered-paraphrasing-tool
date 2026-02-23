"""
AI-Powered Paraphrasing Tool
Console-based application using LLaMA 2 and modern NLP evaluation metrics.

Author: Rishitha Raj
"""

import torch
import nltk
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sacrebleu
import language_tool_python

nltk.download('punkt')


# ------------------------------
# Load Models
# ------------------------------

def load_llm():

    # model_name = "meta-llama/Llama-2-7b-chat-hf" #Better paraphrasing but slower inference
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" #Poor performance on paraphrasing but quick inference
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        # device_map="auto"
    )

    model.to(device)

    return tokenizer, model

def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def load_grammar_tool():
    return language_tool_python.LanguageTool('en-US')


# ------------------------------
# Paraphrasing Function
# ------------------------------

def paraphrase_text(text, tokenizer, model): # for mistral, change to paraphrase_text_mistral

    messages = [
        {
            "role": "system",
            "content": "You are a precise assistant that rewrites sentences clearly while preserving the original meaning."
        },
        {
            "role": "user",
            "content": f"Rewrite the following sentence in a different way. "
                       f"Preserve the full meaning. "
                       f"Do not add new information. "
                       f"Output only one sentence.\n\nSentence:\n{text}"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt part
    result = decoded[len(prompt):].strip()
    result = result.split("\n")[0].strip('"')

    return result

# def paraphrase_text(text, tokenizer, model): # for tinyllama

#     messages = [
#         {"role": "system", "content": "You are a helpful assistant that rewrites sentences clearly and concisely."},
#         {"role": "user", 
#  "content": f"Rewrite the following sentence while preserving the full meaning. Do not remove important information. Only output one sentence.\n\nSentence:\n{text}"}
#     ]

#     prompt = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )

#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=100,
#             temperature=0.3,
#             top_p=0.9,
#             repetition_penalty=1.2,
#             do_sample=True
#         )

#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Remove original prompt part
#     result = decoded[len(prompt):].strip()
#     result = result.split("\n")[0]

#     return result

# ------------------------------
# Grammar Correction
# ------------------------------

def grammar_check(text, tool):
    matches = tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)
    return corrected


# ------------------------------
# Evaluation
# ------------------------------

def evaluate(original, paraphrased, semantic_model):

    print("\n--- Evaluation ---")

    # BLEU with smoothing
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu(
        [original.split()],
        paraphrased.split(),
        smoothing_function=smooth
    )
    print("BLEU:", round(bleu, 3))

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(original, paraphrased)
    print("ROUGE-1:", round(rouge_scores['rouge1'].fmeasure, 3))
    print("ROUGE-L:", round(rouge_scores['rougeL'].fmeasure, 3))

    # Semantic Similarity
    emb1 = semantic_model.encode(original, convert_to_tensor=True)
    emb2 = semantic_model.encode(paraphrased, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2)
    print("Semantic Similarity:", round(similarity.item(), 3))

    # SacreBLEU
    sacre = sacrebleu.corpus_bleu([paraphrased], [[original]])
    print("SacreBLEU:", round(sacre.score, 2))


# ------------------------------
# Main
# ------------------------------

def main():

    parser = argparse.ArgumentParser(description="AI-powered paraphrasing tool")
    parser.add_argument("--text", type=str, help="Input text to paraphrase")

    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        text = input("Enter text to paraphrase:\n")

    print("\nLoading models...")
    tokenizer, llm_model = load_llm()
    semantic_model = load_embedding_model()
    grammar_tool = load_grammar_tool()

    raw_output = paraphrase_text(text, tokenizer, llm_model)
    final_output = grammar_check(raw_output, grammar_tool)

    print("\nOriginal:\n", text)
    print("\nParaphrased:\n", final_output)

    evaluate(text, final_output, semantic_model)
    grammar_tool.close()


if __name__ == "__main__":
    main()