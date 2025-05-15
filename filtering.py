import fasttext
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir="./models")
model = fasttext.load_model(model_path)
lang_codes = {"english": "__label__eng_Latn", "sinhala": "__label__sin_Sinh"}
filtering_stats = {}
#Define length ratio parameters based on NLLB
LENGTH_RATIO_MEAN = 0
LENGTH_RATIO_STD = 0

with open("data/en-si/NLLB.en-si.en") as f1, open("data/en-si/NLLB.en-si.si") as f2: 
    ratios = []
    for line1, line2 in zip(f1, f2):
        line1 = line1.removesuffix("\n")
        line2= line2.removesuffix("\n")
        ratios.append(float(len(line1)/len(line2)))
    import statistics 
    LENGTH_RATIO_MEAN = statistics.fmean(ratios)
    LEGNTH_RATIO_STD = statistics.stdev(ratios)


#Check if the lengths of the sentence pairs match:
def check_lengths(df, lang1, lang2, z_thresh=2.5):
    import numpy as np
    ratios = df[f"{lang1}"].apply(lambda x: len(x.split())) / df[f"{lang2}"].apply(lambda x: len(x.split()) if len(x.split()) > 0 else 1)
    log_ratios = np.log(ratios)
    mean = log_ratios.mean()
    std = log_ratios.std()
    z_scores = (log_ratios - mean) / std
    return df[np.abs(z_scores) <= z_thresh].reset_index(drop=True)

#Return true if the sentence belongs to the specified language
def check_language(sentence, language_code, threshold):
    x = model.predict(sentence)
    if str(x[0][0]) == language_code and float(x[1][0]) >= threshold:
        return True 
    else:
        return False
 
#Convert Moses format files into a pandas DataFrame
def moses_to_df(file1, file2, lang1, lang2):
    lang1_lines = []
    lang2_lines = []

    #Read Moses files into a Pandas DataFrame
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        for line1, line2 in zip(f1, f2): 
            line1 = line1.removesuffix("\n")
            line2= line2.removesuffix("\n")
            lang1_lines.append(line1)
            lang2_lines.append(line2)
    import pandas as pd
    df = pd.DataFrame({lang1: lang1_lines, lang2: lang2_lines})
    filtering_stats["Raw corpus"] = df.shape[0]
    #Remove duplicated sentence pairs 
    df.drop_duplicates(inplace=True, ignore_index=True)
    filtering_stats["After dropping duplicates"] = df.shape[0]
    #Remove pairs where the ratios of words per sentence is too unlikely
    df = check_lengths(df, "english", "sinhala")
    filtering_stats["After removing length based outliers"] = df.shape[0]
    #Remove pairs where one of the sentences is in the wrong language
    lang1_mask = df[f"{lang1}"].apply(check_language, args=(lang_codes[f"{lang1}"], 0.5))
    lang2_mask = df[f"{lang2}"].apply(check_language, args=(lang_codes[f"{lang2}"], 0.5))
    df = df[lang1_mask & lang2_mask].reset_index(drop=True)
    filtering_stats["After performing language identification"] = df.shape[0]
    return df

#Convert a list of sentences into their multilingual embeddings according to the given model
def to_multilingual_embedding(language, sentences, model):
    import torch
    device = ""
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
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
def filter(df, mode): 
    df.sort_values("Similarity score", ascending=False, inplace=True)
    df = df[df["Similarity score"] >= mode]
    return df

def main():
    import argparse 
    parser = argparse.ArgumentParser("filtering.py")
    parser.add_argument("--files", "-f", type=str, nargs="+")
    parser.add_argument("--langs", "-l", type=str, nargs="+")
    parser.add_argument("--output", "-o", type=str)
    # parser.add_argument("--percentile", "-p", type=float)
    parser.add_argument("--model", "-m", type=str)
    args = parser.parse_args()
    df = moses_to_df(args.files[0], args.files[1], args.langs[0], args.langs[1])
    lang1_embedding = to_multilingual_embedding(args.langs[0], df[args.langs[0]], args.model)
    lang2_embedding = to_multilingual_embedding(args.langs[1], df[args.langs[1]], args.model)
    df["Similarity score"] = find_similarity_score(lang1_embedding, lang2_embedding)
    import numpy as np
    hist, bin_edges = np.histogram(list(df["Similarity score"]), bins=100)
    peak_bin_index = np.argmax(hist)
    mode = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index + 1]) / 2
    df = filter(df, mode)
    filtering_stats["After filtering based on similarity scores"] = df.shape[0]
    import align_source_target as ast
    source_lines = df[args.langs[0]]
    target_lines = df[args.langs[1]]
    margin_lines = list(df.index)
    src_file = ""
    trg_file=""
    mrg_file = ""
    for id, sentence in zip(margin_lines, source_lines):
        src_file = src_file + f"{id}" + "\t" + sentence + "\n"
    for id, sentence in zip(margin_lines, target_lines):
        trg_file = trg_file + f"{id}" + "\t" + sentence + "\n"
    for id, sentence in zip(margin_lines, source_lines):
        mrg_file = mrg_file + f"{id}" + "\t" + f"{id}" + "\n"
    src_file_dict = ast.text_to_dict(src_file)
    trg_file_dict = ast.text_to_dict(trg_file)
    margin_train_file = mrg_file
    split_margin_train = margin_train_file.split('\n')[:-1]
    align_list_train = ast.align_source_target(split_margin_train, src_file_dict, trg_file_dict) 
    align_rate_train_file = ast.align_rate_file(split_margin_train, align_list_train, src_file_dict, trg_file_dict) 
    alignment_score = []
    for line in align_rate_train_file:
        split_align = line.split("\t")
        alignment_score.append(float(split_align[2]))
    df["Alignment score"] = alignment_score
    df = df[df["Alignment score"] >= 0.3]
    filtering_stats["After filtering based on word alignment"] = df.shape[0]
    df.to_csv(args.output, sep="\t")
    for item in filtering_stats.keys():
        print(item + f": {filtering_stats[item]}\n")
    print(mode)
if __name__ == "__main__":
    main()