import streamlit as st
from PIL import Image
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import BlipProcessor, BlipForConditionalGeneration

# Device
device = "cpu"  # Colab safe

# ---------------- Load models ----------------
st.write("🧠 Loading pre-trained VQA and Captioning models... Please wait a moment.")

# VQA model
vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vqa_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)
vqa_model.eval()

# Captioning model (BLIP)
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
caption_model.eval()

st.success("✅ Models loaded successfully!")

# ---------------- Page title ----------------
st.title("🖼️ Visual Question Answering & Captioning Demo")
st.write("Upload an image, ask questions or get a full description of the image!")

# ---------------- Session state ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Upload image ----------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Layout: 2 columns
    col1, col2 = st.columns(2)

    # ---------------- Left column: Image ----------------
    with col1:
        st.subheader("🖼️ Uploaded Image")
        st.image(image, use_container_width=True)

    # ---------------- Right column: Interactions ----------------
    with col2:
        st.subheader("❓ Ask a Question or Describe Image")

        # Example questions
        example_questions = [
            "What is the man doing?",
            "What color is the car?",
            "Is there an animal in the image?",
            "What is in the sky?"
        ]
        selected_question = st.selectbox("Or select an example question:", example_questions)
        custom_question = st.text_input("Or type your own question here:")

        question = custom_question if custom_question.strip() != "" else selected_question

        # Use a form for clean buttons
        with st.form("interaction_form"):
            get_answer = st.form_submit_button("Get Answer (VQA)")
            describe_image = st.form_submit_button("Describe Image")

            # ---------------- VQA Prediction ----------------
            if get_answer:
                with st.spinner("Running VQA model..."):
                    encoding = vqa_processor(image, question, return_tensors="pt").to(device)
                    outputs = vqa_model(**encoding)
                    logits = outputs.logits
                    answer_idx = logits.argmax(-1).item()
                    answer = vqa_model.config.id2label[answer_idx]

                # Add to history
                st.session_state.history.append({"question": question, "answer": answer})
                st.subheader("💡 VQA Answer:")
                st.success(answer)

            # ---------------- Image Captioning ----------------
            if describe_image:
                with st.spinner("Generating image description..."):
                    inputs = caption_processor(images=image, return_tensors="pt").to(device)
                    out = caption_model.generate(**inputs)
                    description = caption_processor.decode(out[0], skip_special_tokens=True)

                # Add to history
                st.session_state.history.append({"question": "Describe Image", "answer": description})
                st.subheader("📝 Image Description:")
                st.success(description)

        # ---------------- History ----------------
        if st.session_state.history:
            st.subheader("🗂️ Previous Questions & Answers (Last 5)")
            for item in reversed(st.session_state.history[-5:]):
                st.write(f"❓ {item['question']}")
                st.write(f"💡 {item['answer']}")
                st.markdown("---")
