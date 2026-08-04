import random
import urllib.parse
import numpy as np
import nltk
from nltk.stem import LancasterStemmer
from sklearn.neural_network import MLPClassifier
import streamlit as st

# --- 0. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Mshirika - Stima SACCO Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Mshirika - Stima SACCO Virtual Assistant")
st.caption("Your 24/7 assistant for Stima SACCO membership, loans, dividends, and mobile banking.")

# --- 1. CACHED NLTK ASSETS ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

download_nltk_data()
stemmer = LancasterStemmer()

# --- 2. KNOWLEDGE BASE DATASET ---
intents_data = {
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Habari", "How are you", "Is anyone there?", "Good day", "Mambo"],
      "responses": [
        "Hello! I'm Mshirika, your Stima SACCO virtual assistant. I can help with membership, loans, accounts, dividends, and mobile banking questions. How can I assist you today?"
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
      "tag": "name",
      "patterns": ["what is your name", "who are you", "what should I call you"],
      "responses": [
        "I'm Mshirika, Stima SACCO's virtual assistant, here to answer common member questions and connect you to customer care when needed."
      ]
    },
    {
      "tag": "membership_requirements",
      "patterns": [
        "how do I become a member", "how do I join Stima Sacco", "what do I need to join",
        "can I join as diaspora", "documents needed for membership", "how to register as a member",
        "advise on how to open an account", "open account", "membership requirements"
      ],
      "responses": [
        "To join Stima SACCO you'll need: (1) a completed membership application form, (2) a copy of your National ID or valid passport, (3) a minimum Share Capital contribution of KSh 25,000 (payable in installments), and (4) monthly Alpha Deposits of at least KSh 1,000 to keep your account active. Open to individuals, chamas, corporates, and diaspora. Download the form: https://www.stima-sacco.com/downloads/loan-application-forms/ or email diasporabanking@stima-sacco.com for diaspora queries."
      ]
    },
    {
      "tag": "account_types",
      "patterns": [
        "what is Share Capital", "what is Alpha Deposits", "what is a Prime account",
        "what is FOSA account", "difference between share capital and alpha deposits", "what accounts do you offer"
      ],
      "responses": [
        "Stima SACCO has three core account types:\n"
        "1. **Share Capital Account:** Equity/co-ownership stake - non-withdrawable, but transferable/sellable to another member upon exit.\n"
        "2. **Alpha Deposits Account:** Monthly savings used as loan collateral - refunded in full upon exiting and clearing liabilities.\n"
        "3. **Prime/FOSA Account:** Transactional account for salaries, ATM withdrawals, over-the-counter transactions, and receiving dividends."
      ]
    },
    {
      "tag": "loan_eligibility",
      "patterns": [
        "am I eligible for a loan", "how long before I can borrow", "how many months before I can get a loan",
        "loan requirements", "can I apply for a loan now"
      ],
      "responses": [
        "You need to be a member for at least 3 months (individuals) or 6 months (corporate members) and meet product-specific requirements (active deposits, payslips/bank statements, ID copy, and KRA PIN). View products: https://www.stima-sacco.com/credit-products/"
      ]
    },
    {
      "tag": "loan_multiplier_security",
      "patterns": [
        "how much can I borrow", "can I borrow 4 times my deposits", "loan multiplier",
        "do I need a guarantor for a big loan", "why do I need security for my loan",
        "can I use land as loan security", "can I use my car logbook for a loan"
      ],
      "responses": [
        "You can generally borrow up to 3–4 times your Alpha Deposits, subject to security (guarantors, self-guarantee, land title deeds, motor vehicle logbooks, or fixed deposits). The SACCO determines final eligibility based on your risk profile."
      ]
    },
    {
      "tag": "loan_products",
      "patterns": [
        "what loan products do you have", "what types of loans are available", "do you have business loans",
        "do you have mortgage loans", "do you have Islamic finance loans", "short term loan options", "emergency loan", "loans"
      ],
      "responses": [
        "Stima SACCO offers Short-term loans (Salary Advance, Emergency, School fees), Long-term loans (Normal, Super, Flex, Mwangaza), Business/Asset Finance, Mortgages (KMRC), and Sharia-compliant options (Mudarabah, Musharaka). Download forms: https://www.stima-sacco.com/downloads/loan-application-forms/"
      ]
    },
    {
      "tag": "dividends",
      "patterns": ["when are dividends paid", "how much dividends will I get", "when is the AGM", "interest rebate", "how are the dividends"],
      "responses": [
        "Dividends (on Share Capital) and interest rebates (on Alpha Deposits) are paid annually after the Annual General Meeting (AGM) - typically late February to March."
      ]
    },
    {
      "tag": "dividends_discounting",
      "patterns": ["can I get an advance on my dividends", "dividends discounting", "advance against dividends"],
      "responses": [
        "Yes, you can access up to 50% of your estimated dividends/rebates in advance based on the prior year's payout at an interest rate of 4% per month. Apply via M-Pawa/M-Stima, internet banking, or your nearest branch."
      ]
    },
    {
      "tag": "mobile_banking",
      "patterns": ["how do I check my balance", "how do I use M-Stima", "USSD code", "mobile app", "how do I register for M-Stima"],
      "responses": [
        "Dial ***492#** for USSD banking or use the **M-Stima App** (Android & iOS). For setup assistance, contact WhatsApp: 0703024001 or customercare@stima-sacco.com."
      ]
    },
    {
      "tag": "paybill_deposits",
      "patterns": ["how do I deposit money", "paybill number", "how do I pay my monthly deposit via mpesa", "how do I top up my share capital"],
      "responses": [
        "Use **M-PESA Paybill 823244**:\n"
        "* **Alpha Deposits:** `802` + `7-digit Member Number` + `00`\n"
        "* **Share Capital:** `800` + `7-digit Member Number` + `00`\n"
        "* **Prime/FOSA Account:** `801` + `7-digit Member Number` + `00`"
      ]
    },
    {
      "tag": "exit_withdrawal",
      "patterns": ["how do I leave Stima Sacco", "how do I withdraw my membership", "can I close my account"],
      "responses": [
        "To exit, submit a formal 60-day written notice. Alpha Deposits are refunded in full after settling outstanding liabilities. Share Capital is non-withdrawable but can be transferred or sold to an active member."
      ]
    },
    {
      "tag": "guarantor_rules",
      "patterns": ["can I withdraw if I guaranteed someone", "I am a guarantor can I leave", "guarantor liability"],
      "responses": [
        "You cannot withdraw or close your account while actively guaranteeing another member's loan unless the borrower clears the balance or replaces you with an eligible guarantor."
      ]
    },
    {
      "tag": "branch_locator",
      "patterns": ["where is your nearest branch", "branches", "office locations", "where is your head office"],
      "responses": [
        "Head Office: Stima Sacco Plaza, Mushembi Road, Parklands, Nairobi. Other branches: Nairobi CBD (Kawi Centre), Mombasa, Kisumu, Nakuru, Olkaria, Eldoret, and Embu. Full locator: https://www.stima-sacco.com/branches/"
      ]
    },
    {
      "tag": "complaint_escalation",
      "patterns": ["I want to file a complaint", "I have an issue with my account", "talk to a human", "connect me to customer care"],
      "responses": [
        "I'm escalating this to Customer Care: Email customercare@stima-sacco.com | Phone: 0703024000 / 0703024024 | WhatsApp: 0703024001."
      ]
    }
  ]
}

# --- 3. MODEL TRAINING & CACHING ---
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

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            bag.append(1 if w in wrds else 0)

        training.append(bag)
        output.append(docs_y[x])

    training = np.array(training)
    output = np.array(output)

    # Neural Network (MLP) Classifier via Scikit-Learn
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    model.fit(training, output)
    
    return model, words, labels

model, words, labels = build_and_train_model()

# --- 4. HELPER FUNCTIONS ---
def clean_user_input(text):
    """Strips leading conversational greetings if the message contains an actual query."""
    greetings = ["hi", "hello", "hey", "habari", "mambo", "good day", "good morning", "good afternoon"]
    words_list = text.strip().split()
    
    if len(words_list) > 1:
        first_word = words_list[0].lower().strip(".,!?")
        if first_word in greetings:
            return " ".join(words_list[1:])
            
    return text

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
    subject = "Escalated Member Inquiry - Mshirika Virtual Assistant"
    body = (
        f"Dear Stima SACCO Customer Care Team,\n\n"
        f"I need support with the following query:\n\n"
        f"\"{user_query}\"\n\n"
        f"Member Details:\n"
        f"Name: \n"
        f"Member No: \n"
        f"Phone: \n\n"
        f"Kind regards,"
    )
    mailto_url = f"mailto:{recipient}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
    return mailto_url, recipient

# --- 5. STREAMLIT CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process new user input
if prompt := st.chat_input("Ask about deposits, loans, dividends, USSD..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Clean leading greetings from multi-word prompts
    cleaned_prompt = clean_user_input(prompt)

    # Predict intent & probability on cleaned prompt
    probs = model.predict_proba(bag_of_words(cleaned_prompt, words))[0]
    max_idx = np.argmax(probs)
    tag = model.classes_[max_idx]
    confidence = probs[max_idx]

    # Confidence threshold evaluation
    if confidence > 0.40 and tag not in ["complaint_escalation"]:
        for tg in intents_data["intents"]:
            if tg['tag'] == tag:
                bot_reply = random.choice(tg['responses'])
    else:
        mailto_link, email_addr = generate_escalation_link(prompt)
        bot_reply = (
            f"I couldn't confidently process your request. I can escalate this directly to our Customer Support team.\n\n"
            f"👉 **Direct Email:** `{email_addr}`\n\n"
            f"👉 **[Click here to open pre-filled email]({mailto_link})**"
        )

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})