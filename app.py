import random
import urllib.parse
import numpy as np
import nltk
from nltk.stem import LancasterStemmer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import streamlit as st

# Download NLTK assets
nltk.download('punkt')
nltk.download('punkt_tab')
stemmer = LancasterStemmer()

# Streamlit Page Config
st.set_page_config(page_title="Mshirika - Stima SACCO Assistant", page_icon="🤖")
st.title("🤖 Mshirika - Stima SACCO Virtual Assistant")
st.caption("Ask questions about membership, loans, dividends, and mobile banking.")

# --- 1. KNOWLEDGE BASE DATASET ---
intents_data = {
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Habari", "How are you", "Is anyone there?", "Good day", "Mambo"],
      "responses": [
        "Hello! I'm Mshirika, your Stima SACCO assistant. I can help with membership, loans, accounts, dividends, and mobile banking questions. How can I help you today?"
      ]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you later", "Goodbye", "I am leaving", "Asante, that's all"],
      "responses": [
        "You're welcome! Reach out anytime you have a question. Karibu tena!"
      ]
    },
    {
      "tag": "membership_requirements",
      "patterns": [
        "how do I become a member", "how do I join Stima Sacco", "what do I need to join",
        "can I join as diaspora", "documents needed for membership", "how to register as a member"
      ],
      "responses": [
        "To join Stima SACCO you'll need: (1) a completed application form, (2) National ID/Passport copy, (3) KSh 25,000 Share Capital (payable in installments), and (4) monthly Alpha Deposits of at least KSh 1,000. Open to individuals, chamas, corporates, and diaspora (diasporabanking@stima-sacco.com)."
      ]
    },
    {
      "tag": "account_types",
      "patterns": [
        "what is Share Capital", "what is Alpha Deposits", "what is a Prime account",
        "what is FOSA account", "difference between share capital and alpha deposits"
      ],
      "responses": [
        "Core accounts: (1) Share Capital (equity/co-ownership stake - non-withdrawable, transferable upon exit), (2) Alpha Deposits (monthly savings used as loan collateral - refunded upon exit), and (3) Prime/FOSA Account (transactional account for salary, ATMs, and dividend payouts)."
      ]
    },
    {
      "tag": "paybill_deposits",
      "patterns": [
        "how do I deposit money", "paybill number", "how do I pay my monthly deposit via mpesa", "how do I top up my share capital"
      ],
      "responses": [
        "Use Paybill 823244. For Alpha Deposits: 802 + 7-digit member number + 00. For Share Capital: 800 + 7-digit member number + 00. For Prime (FOSA): 801 + 7-digit member number + 00."
      ]
    },
    {
      "tag": "mobile_banking",
      "patterns": ["how do I check my balance", "how do I use M-Stima", "USSD code", "mobile app"],
      "responses": [
        "Dial *492# for USSD banking or use the M-Stima app (Android/iOS). For support, contact WhatsApp: 0703024001 or email customercare@stima-sacco.com."
      ]
    },
    {
      "tag": "complaint_escalation",
      "patterns": ["I want to file a complaint", "talk to a human", "connect me to customer care"],
      "responses": ["Let me connect you with Customer Care: customercare@stima-sacco.com | 0703024000 / 0703024024."]
    }
  ]
}

# --- 2. TRAIN MODEL (CACHED FOR PERFORMANCE) ---
@st.cache_resource
def build_and_train_model():
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in ["?", "!", "."]]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            bag.append(1 if w in wrds else 0)

        output_row = list(out_empty)
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    model = Sequential([
        Dense(32, input_shape=(len(training[0]),), activation='relu'),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(len(output[0]), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(training, output, epochs=300, batch_size=8, verbose=0)
    
    return model, words, labels

model, words, labels = build_and_train_model()

# --- 3. HELPER FUNCTIONS ---
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array([bag])

def generate_escalation_link(user_query):
    recipient = "customercare@stima-sacco.com"
    subject = "Escalated Member Inquiry - Stima SACCO Support"
    body = f"Dear Stima SACCO Customer Care Team,\n\nI need assistance regarding: \"{user_query}\"\n\nKind regards,"
    return f"mailto:{recipient}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}", recipient

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you with Stima SACCO today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    results = model.predict(bag_of_words(prompt, words), verbose=0)[0]
    results_index = np.argmax(results)
    tag = labels[results_index]
    confidence = results[results_index]

    if confidence > 0.70 and tag not in ["fallback", "complaint_escalation"]:
        for tg in intents_data["intents"]:
            if tg['tag'] == tag:
                bot_reply = random.choice(tg['responses'])
    else:
        mailto_link, email_addr = generate_escalation_link(prompt)
        bot_reply = (
            f"I am unable to fully process your query right now. Let's escalate this to our Customer Care team.\n\n"
            f"👉 **Direct Email:** `{email_addr}`\n\n"
            f"👉 **[Click here to send pre-filled email]({mailto_link})**"
        )

    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
