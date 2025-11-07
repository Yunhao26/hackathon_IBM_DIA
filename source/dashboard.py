import streamlit as st
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoTokenizer

# Load the trained model
# model = joblib.load(r"C:\Users\adrib\Documents\ESILV\A5\BI Pipeline\model_9d1d2b6d-b501-4495-bc8b-67e1b85d734d.pickle\P2.pickle")

# Default values (e.g., from your dataset)
DEFAULT_VALUES = {
    "prompt_token_length": 50,
    "response_token_length": 100,
    "word_count": 200,
    "sentence_count": 10,
    "hardware_type": "laptop",
    "model_name": "Llama",
    "energy_mix": 50  # kgCO₂e/kWh (France)
}

# Streamlit UI
st.title("LLM Energy Consumption Dashboard")

# Input section FORMS
st.header("Input Query Characteristics")
model_name = st.selectbox("Model", ["gemma:2b", "gemma:7b", "codellama:7b","llama3","codellama","llama3:70b"], index=0)
prompt = st.text_area("Prompt", height=150)
prompt_language = st.selectbox("Prompt Language", ["french", "english"], index=0)
response = st.text_area("Response", height=150)
energy_mix = st.number_input("Energy Mix (gCO₂e/kWh) ex: France - 50; Allemagne - 350; UK - 200; USA - 400; Chine - 500; Japon - 450", min_value = 0, value=DEFAULT_VALUES["energy_mix"])
total_duration = st.number_input("Total Duration (seconds)", min_value=1, value=60)

def calculate_prompt_features(prompt):
    """
    Calcule les features suivantes à partir d'un prompt :
    - word_count
    - sentence_count
    - avg_word_length
    - word_diversity
    - unique_word_count
    - avg_sentence_length
    - reading_time (en secondes)
    """
    features = {}

    # 1. word_count : Nombre total de mots
    words = word_tokenize(prompt, language=prompt_language)  # ou 'english' selon la langue
    features['word_count'] = len(words)

    # 2. sentence_count : Nombre de phrases
    sentences = sent_tokenize(prompt, language=prompt_language)  # ou 'english'
    features['sentence_count'] = len(sentences)

    # 3. avg_word_length : Longueur moyenne des mots (en caractères)
    features['avg_word_length'] = sum(len(word) for word in words) / features['word_count'] if features['word_count'] > 0 else 0

    # 4. unique_word_count : Nombre de mots uniques
    features['unique_word_count'] = len(set(words))

    # 5. word_diversity : Rapport mots uniques / mots totaux
    features['word_diversity'] = features['unique_word_count'] / features['word_count'] if features['word_count'] > 0 else 0

    # 6. avg_sentence_length : Longueur moyenne des phrases (en mots)
    features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0

    # 7. reading_time : Temps de lecture estimé (en secondes, 200 mots/minute)
    features['reading_time'] = (features['word_count'] / 200) * 60

    return features



# Dictionnaire pour mapper les noms de modèles aux identifiants Hugging Face
MODEL_TOKENIZERS = {
    "codellama:7b": "codellama/CodeLlama-7b-hf",
    "gemma:2b": "google/gemma-2b",
    "gemma:7b": "google/gemma-7b",
    "llama3:70b": "meta-llama/Meta-Llama-3-70B",  # Remplace par le vrai nom si différent
    "llama3": "meta-llama/Meta-Llama-3-8B",      # Exemple pour Llama3 (8B)
    "codellama:70b": "codellama/CodeLlama-70b-hf",
    "codellama": "codellama/CodeLlama-7b-hf"     # Par défaut, utilise CodeLlama-7B
}

from huggingface_hub import login

# Replace with your Hugging Face access token
HF_TOKEN = "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"
login(token=HF_TOKEN)

# Cache pour stocker les tokenizers (optimisation)
tokenizer_cache = {}

def get_prompt_token_length(prompt: str, model_name: str) -> int:
    """
    Calcule le nombre de tokens pour un prompt donné et un modèle spécifique.

    Args:
        prompt (str): Le texte du prompt.
        model_name (str): Le nom du modèle (ex: "codellama:7b", "gemma:2b").

    Returns:
        int: Le nombre de tokens dans le prompt.
    """
    # Vérifie si le modèle est supporté
    if model_name not in MODEL_TOKENIZERS:
        raise ValueError(f"Modèle non supporté : {model_name}. Modèles disponibles : {list(MODEL_TOKENIZERS.keys())}")

    # Charge le tokenizer (avec mise en cache pour éviter de recharger)
    if model_name not in tokenizer_cache:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZERS[model_name])
        tokenizer_cache[model_name] = tokenizer
    else:
        tokenizer = tokenizer_cache[model_name]

    # Tokenize le prompt et retourne la longueur
    tokens = tokenizer.encode(prompt)
    return len(tokens)

def get_response_token_length(response: str, model_name: str) -> int:
    """
    Calcule le nombre de tokens pour une réponse donné et un modèle spécifique.

    Args:
        response (str): Le texte de la réponse.
        model_name (str): Le nom du modèle (ex: "codellama:7b", "gemma:2b").

    Returns:
        int: Le nombre de tokens dans la réponse.
    """
    # Vérifie si le modèle est supporté
    if model_name not in MODEL_TOKENIZERS:
        raise ValueError(f"Modèle non supporté : {model_name}. Modèles disponibles : {list(MODEL_TOKENIZERS.keys())}")

    # Charge le tokenizer (avec mise en cache pour éviter de recharger)
    if model_name not in tokenizer_cache:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZERS[model_name])
        tokenizer_cache[model_name] = tokenizer
    else:
        tokenizer = tokenizer_cache[model_name]

    # Tokenize le prompt et retourne la longueur
    tokens = tokenizer.encode(response)
    return len(tokens)

if prompt:
    features = calculate_prompt_features(prompt)
    prompt_token_length = get_prompt_token_length(prompt, model_name)
    response_token_length = get_response_token_length(response, model_name)
    st.subheader("Features calculées :")
    st.write(f"- **Nombre de mots** : {features['word_count']}")
    st.write(f"- **Nombre de mots uniques** : {features['unique_word_count']}")
    st.write(f"- **Longueur moyenne des mots** : {features['avg_word_length']:.2f} caractères")
    st.write(f"- **Nombre de phrases** : {features['sentence_count']}")
    st.write(f"- **Longueur du prompt en tokens** : {prompt_token_length} tokens")
    st.write(f"- **Longueur de la response en tokens** : {response_token_length} tokens")

    data = [
    model_name,
    total_duration,
    prompt_token_length,
    features['unique_word_count'], 
    response_token_length,
    features['avg_word_length'], 
    features['sentence_count']
]
    # Predict energy consumption
    coeffs = {
        "gemma:2b": 0.08,
        "gemma:7b": 0.12,
        "codellama:7b": 0.15,
        "llama3": 0.10,
        "codellama": 0.09,
        "llama3:70b": 0.20
    }
    model_coeff = coeffs.get(model_name, 0.1)  # Default coefficient if model_name is not in the dictionary

    base_energy_kwh = 10
    # Calcul des prédictions pour chaque modèle
    energies_kwh = [base_energy_kwh * coeffs[model] for model in coeffs.keys()]
    co2eqs = [energy * energy_mix/1000 for energy in energies_kwh]

    # Création d'un DataFrame pour les graphiques
    df = pd.DataFrame({
        "Modèle": coeffs.keys(),
        "Énergie (kWh)": energies_kwh,
        "CO₂e (kg)": co2eqs
    })

    # Highlight the selected model
    df["Highlight"] = df["Modèle"] == model_name

    # Affichage des graphiques
    st.subheader("Comparaison des modèles")

    # Graphique 1 : Énergie (kWh)
    fig, ax = plt.subplots()
    sns.barplot(
        data=df, 
        x="Modèle", 
        y="Énergie (kWh)", 
        ax=ax, 
        palette=["gold" if highlight else "blue" for highlight in df["Highlight"]]
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Consommation d'énergie par modèle (kWh)")
    st.pyplot(fig)

    # Graphique 2 : CO₂e (kg)
    fig, ax = plt.subplots()
    sns.barplot(
        data=df, 
        x="Modèle", 
        y="CO₂e (kg)", 
        ax=ax, 
        palette=["gold" if highlight else "blue" for highlight in df["Highlight"]]
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Émissions de CO₂e par modèle (kg)")
    st.pyplot(fig)

    # Affichage des données sous forme de tableau
    st.subheader("Données comparatives")
    st.dataframe(df)
