import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch

@st.cache_resource
def load_model():
    model_name = "prajjwal1/bert-tiny-mnli"   # super lightweight
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("📝 Contract Clause Verifier (Tiny NLI)")

premise = st.text_area("Contract Clause (Premise)", 
                       "The tenant shall maintain valid insurance during the lease term.")
hypothesis = st.text_area("Summary Claim (Hypothesis)", 
                          "Tenant must provide insurance coverage.")

if st.button("Verify"):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze()

    labels = ["Contradiction", "Neutral", "Entailment"]
    result = {labels[i]: float(probs[i]) for i in range(len(labels))}
    st.write("### Results")
    st.json(result)
    st.success(f"Predicted: {labels[probs.argmax().item()]} ✅")
