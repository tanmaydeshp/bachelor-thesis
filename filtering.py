import fasttext
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir="./models")
model = fasttext.load_model(model_path)
lang_codes = {"english": "__label__eng_Latn", "sinhala": "__label__sin_Sinh"}

#Check if the lengths of the sentence pairs match:
def check_lengths(sentence1, sentence2):
    words1 = len(sentence1.split())
    words2 = len(sentence2.split())
    if words1 >= 5 * words2 or words2 >= 5 * words1: 
        return False 
    else:
        return True

#Return true if the sentence belongs to the specified language
def check_language(sentence, language_code):
    x = model.predict(sentence)
    if str(x[0][0]) == language_code and float(x[1][0]) >= 0.5:
        return True 
    else:
        return False
 
#Convert Moses format files into a pandas DataFrame
def moses_to_df(file1, file2, lang1, lang2):
    lang1_lines = []
    lang2_lines = []
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        for line1, line2 in zip(f1, f2): 
            line1 = line1.removesuffix("\n")
            line2= line2.removesuffix("\n")
            if not(check_lengths(line1, line2)):
                continue
            if (check_language(line1, lang_codes[lang1]) and check_language(line2, lang_codes[lang2])):
                lang1_lines.append(line1)
                lang2_lines.append(line2)
    import pandas as pd
    df = pd.DataFrame({lang1: lang1_lines, lang2: lang2_lines})
    df.drop_duplicates(inplace=True, ignore_index=True)
    return df

#Convert a list of sentences into their multilingual embeddings according to the given model
def to_multilingual_embedding(language, sentences, model):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model.lower() == "labse":
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('sentence-transformers/LaBSE', device=device)
        embedding = encoder.encode(sentences, device=device)
    if model.lower() == "laser":
        import argparse
        torch.serialization.add_safe_globals([argparse.Namespace])
        from laser_encoders import LaserEncoderPipeline
        encoder = LaserEncoderPipeline(lang=language)
        embedding = encoder.encode_sentences(sentences)
    return embedding

#Find similarity scores for sentence pairs using cosine similarity
def find_similarity_score(embeddings1, embeddings2): 
    import statistics
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings1, embeddings2)
    similarity_scores = [statistics.fmean(vector) for vector in similarities]
    # similarity_scores = [float(vector.sum()) for vector in similarities]
    return similarity_scores

#Given a pandas DataFrame, filter best x percent of sentence pairs and store the results in a .tsv file
def filter_top_percentile(df, percentile, tsv_path): 
    df.sort_values("Similarity score", ascending=False, inplace=True)
    df[df["Similarity score"] >= df["Similarity score"].quantile(percentile)].to_csv(sep="\t", path_or_buf=tsv_path)

def main():
    import argparse 
    parser = argparse.ArgumentParser("filtering.py")
    parser.add_argument("--files", "-f", type=str, nargs="+")
    parser.add_argument("--langs", "-l", type=str, nargs="+")
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--percentile", "-p", type=float)
    parser.add_argument("--model", "-m", type=str)
    args = parser.parse_args()
    df = moses_to_df(args.files[0], args.files[1], args.langs[0], args.langs[1])
    lang1_embedding = to_multilingual_embedding(args.langs[0], df[args.langs[0]], args.model)
    lang2_embedding = to_multilingual_embedding(args.langs[1], df[args.langs[1]], args.model)
    df["Similarity score"] = find_similarity_score(lang1_embedding, lang2_embedding)
    filter_top_percentile(df, args.percentile, args.output)

if __name__ == "__main__":
    main()