import numpy as np 
from langdetect import detect
from keybert import KeyBERT 
from transformers import AutoModel, AutoModelForSequenceClassification

def extract(corpus, nbre_mots=10, taille_mots = 2, diversity= 0.7):
   

    langue = detect(corpus)

    def extract_keywords(text):
            keywords =  kw_model.extract_keywords(corpus, keyphrase_ngram_range=(taille_mots, taille_mots),
                                                  use_mmr = True,
                                                  top_n=nbre_mots, diversity = diversity)
            return keywords

    if langue == "en":
        kw_model = KeyBERT(model=AutoModel.from_pretrained('intfloat/e5-base-v2'))
        extract_words = extract_keywords(corpus)
    else:

        kw_model = KeyBERT(model= AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection") )
        extract_words = extract_keywords(corpus)

    return extract_words

# def decorateur(corpus, test_keybert):
#     def test():
#         taille_mots = [1, 2, 3]
#         diversity = [0.5, 0.7, 1]
#         return test_keybert(taille_mots, diversity, corpus)
#     return test


# @decorateur
# def test_keybert(taille_mots, diversity, doc):

#     best_value, best = (0, 0), 0
#     for i in taille_mots:
#         for j in diversity:
#             test = extract(doc, taille_mots=i, nbre_mots = 3, diversity = j)[i][1]
#             if np.mean(test) > best:
#                 best = np.mean(test)
#                 best_value = (i, j)
#     return best_value


text = "La plage ensoleillée est bordée de palmiers majestueux, tandis que les vagues douces caressent le sable blanc. Les surfeurs habiles glissent gracieusement sur l'eau cristalline, capturant l'énergie de l'océan. Les enfants construisent des châteaux de sable tandis que les mouettes volent au-dessus. Le parfum salé de l'air marin remplit mes poumons, créant une sensation de liberté et de bonheur"

print(extract(text, nbre_mots=5))