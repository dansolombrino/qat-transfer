from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model_name = "google/embeddinggemma-300m"
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    # model_name = "google-bert/bert-base-uncased"

    st_model = SentenceTransformer(model_name)
    st_model.to(device=device)

    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
    ]

    query_embeddings = st_model.encode_query(query)
    document_embeddings = st_model.encode_document(documents)
    print("SentenceTransformer:", query_embeddings.shape, document_embeddings.shape)

    # --- AutoModelForSequenceClassification approach ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    hf_model.to(device=device)
    hf_model.eval()

    all_texts = [query] + documents
    inputs = tokenizer(all_texts, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = hf_model(**inputs)

    embeddings = outputs.logits
    hf_query_embedding = embeddings[0]
    hf_document_embeddings = embeddings[1:]
    print("AutoModelForSequenceClassification:", hf_query_embedding.shape, hf_document_embeddings.shape)


if __name__ == "__main__":
    main()