# 🤖 Make-AI-Clone-of-Yourself By Gowtham Varshith

This project creates an **AI clone** of yourself that mimics your conversational style by training a model on your personal WhatsApp chat history. By fine-tuning a large language model (LLM) using custom data, you can have an AI clone that speaks like you, mainly in Hinglish (Hindi + English).

## 🎯 Motivation

Inspired by an Instagram reel, this project was designed to fine-tune an LLM, such as **LLaMA** or **Mixtral**, on personal WhatsApp chat data. This ensures privacy and avoids sending sensitive information to third-party APIs like OpenAI. The project overcomes challenges such as high GPU requirements and converting WhatsApp data into a training-ready format.

---

## 🚀 Google Colab Setup

To get started, follow these steps to run the project in **Google Colab**:

1. **Open the Colab Notebook**:
   [Google Colab Notebook](https://colab.research.google.com/drive/1OGkiAZsYfShY0o8ZphCUuXkmb2Om422X?usp=sharing)

2. **Steps in the Notebook**:
    - **Data Collection**: Pull your WhatsApp chat data.
    - **Data Preparation**: Clean and prepare the chat data for model training.
    - **Data Filtering**: Filter irrelevant data, keeping only conversations that match your style.
    - **Model Training**: Fine-tune the LLM on your chat history.
    - **Inference**: Generate responses using the fine-tuned model.
    - **Model Saving**: Save the trained model for future use.
    - **GGUF Conversion**: Convert the model for compatibility with tools like **Ollama** or **LM Studio**.

---

## 🔧 Code Logic Breakdown

### 1. **Data Collection** (WhatsApp):
Extract chat history and convert it into a dataset format for training the model.

```python
import re

def process_whatsapp_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    messages = re.findall(r'\[.*?\] (.*?): (.*)', data)
    return messages
```
This function processes raw WhatsApp chat files, extracting messages with timestamps and sender information.

### 2. **Data Preparation**:
Prepare the extracted data for model training by filtering and cleaning.

```python
def clean_data(messages):
    cleaned_messages = []
    for message in messages:
        if 'Media omitted' not in message[1]:  # Filter out media messages
            cleaned_messages.append(message)
    return cleaned_messages
```
This function removes irrelevant or non-text data like media messages.

### 3. **Model Fine-tuning**:
Train the model using the cleaned data. You can fine-tune models such as **LLaMA** or **GPT-3.5 Turbo** based on your chat data.

```python
from transformers import Trainer, TrainingArguments

# Model training arguments
training_args = TrainingArguments(
    output_dir='./results', num_train_epochs=3, per_device_train_batch_size=4
)

# Initialize trainer with training data and model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=chat_dataset
)

trainer.train()
```
This code snippet fine-tunes the model on WhatsApp data, adjusting the model's behavior to mimic your chat style.

---

## 💻 Chatting with Fine-Tuned Model Using **Ollama**

### Steps to Chat with the Model:
1. **Download Ollama** from [here](https://ollama.com/download).
2. **Load the Model into Ollama**:
    - Ensure the model file (`unsloth.Q8_0.gguf`) is in the correct folder.
    - Use the command: `ollama create my_model -f Modelfile`.
    - Start chatting with: `ollama run my_model`.

---

## 🤖 Automating WhatsApp Responses

This section allows you to automate your WhatsApp using the fine-tuned model to respond to incoming messages:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Eviltr0N/Make-AI-Clone-of-Yourself.git
   cd Make-AI-Clone-of-Yourself/
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the chat script**:
   ```bash
   python3 chat.py
   ```

4. **Automate WhatsApp**:
   ```bash
   python3 ai_to_whatsapp.py
   ```

Once set up, scan the QR code from WhatsApp and specify the phone number to automate responses.

---

## 🔥 Customizing Model's Responses

You can adjust the **temperature** and **top_k** parameters of the model to make it more creative:

- Open `ai_to_whatsapp.py` and modify:
   ```python
   my_llm = LLM("my_model", 0.3, 50, 128)
   ```

- **Increase temperature** to make responses more creative:
   ```python
   my_llm = LLM("my_model", 0.9, 50, 128)
   ```

---

## 💡 Future Enhancements

- **Multimodal Capabilities**: Extend the model to handle images and videos.
- **Agent Pipeline**: Add an oversight mechanism with another model (like **LLaMA 3.1**) to validate responses before sending them.

---

## 📞 Contact Information

For further questions or collaboration, feel free to reach out:

- **GitHub**: [Gowtham Varshith](https://github.com/Gowtham-Varshith)
- **Email**: gowthamb461@gmail.com

---

## 📝 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

