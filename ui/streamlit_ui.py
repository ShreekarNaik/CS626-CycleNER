import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

st.title("Named Entity Recognition for Kannada Text")

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

tokenizer, model = load_model()

text_input = st.text_area("Enter the text here:")
if st.button("Extract Entities"):
    if text_input.strip():
        input_ids = tokenizer.encode(text_input, return_tensors="pt")
        outputs = model.generate(input_ids)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success(f"Extracted Entities: {decoded_output}")
    else:
        st.error("Please enter valid text.")
