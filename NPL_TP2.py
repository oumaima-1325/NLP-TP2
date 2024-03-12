from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Le texte donné
text = """
Morocco and Marrakech: A Tapestry of Tradition and Modernity Morocco, located at the crossroads of Europe and Africa, is a country drenched in history, mystery, and cultural richness. A testament to the ancient civilizations that once flourished here, this North African kingdom boasts a unique blend of Arab, Berber, and European influences. At the heart of Morocco's rich tapestry lies Marrakech, one of its four imperial cities and a vibrant epicenter of tradition and modernity. Geographical Significance Morocco is bordered by the Atlantic Ocean to the west, the Mediterranean Sea to the north, Algeria to the east and southeast, and the vast Sahara desert to the south. Its strategic location has historically made it a sought-after territory and a melting pot of cultures, religions, and trade routes. Marrakech: The Red City Marrakech, often referred to as "The Red City" due to its distinctive red-hued buildings, stands against the backdrop of the snow-capped Atlas Mountains. Established in the 11th century, it has remained a crucial political, economic, and cultural center of Morocco. Journey through the Medina Marrakech's old town, the Medina, is a UNESCO World Heritage site and a labyrinthine maze of narrow alleys, bustling souks, and historical landmarks. The Djemaa el-Fna Square lies at the heart of the Medina and comes alive every evening with storytellers, musicians, snake charmers, and food stalls offering tantalizing Moroccan delicacies. Palaces and Gardens The city is also home to grand palaces like the Bahia Palace, showcasing intricate Islamic architecture, and the Saadian Tombs, remnants of the Saadian dynasty. The Majorelle Garden, restored by the fashion designer Yves Saint Laurent, is a tranquil oasis of cacti, palm trees, and cobalt blue accents. Modern Marrakech While tradition and history permeate Marrakech, the city is not averse to the modern world. Gueliz, the new town, is brimming with contemporary art galleries, stylish cafes, and chic boutiques, offering a stark contrast to the ancient Medina. Moroccan Cuisine No journey through Morocco and Marrakech would be complete without indulging in the local cuisine. Tagines, couscous, and pastilla are just a few of the many dishes that combine a plethora of flavors and spices like saffron, cumin, and mint. Paired with Moroccan mint tea, the culinary experience is truly unparalleled. In Conclusion Morocco, with Marrakech at its heart, offers travelers an unparalleled journey through time. The convergence of history, culture, architecture, and gastronomy makes it an enthralling destination for those seeking both adventure and reflection. As the Moroccan proverb goes, "He who does not travel does not know the value of men." In the case of Morocco and Marrakech, it's not just the value of men, but also the value of time, tradition, and tales that have spanned centuries.
"""

# Tokenisation des mots et suppression de la ponctuation et des mots vides
tokens = word_tokenize(text.lower())
table = str.maketrans('', '', string.punctuation)
tokens = [w.translate(table) for w in tokens]
tokens = [word for word in tokens if word.isalpha()]
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if not word in stop_words]

# Entraîner le modèle Word2Vec
model = Word2Vec([tokens], vector_size=100, window=5, min_count=1, workers=4)

# 1. Extraire la représentation vectorielle d'un mot
word_representation = model.wv['morocco']
print("Représentation vectorielle du mot 'Morocco':", word_representation)

# 2. Calculer la similarité entre deux mots
similarity = model.wv.similarity('morocco', 'marrakech')
print("Similarité entre 'Morocco' et 'Marrakech':", similarity)

# 3. Extraire les mots contextuels pour un mot central donné
context_words = model.wv.most_similar('morocco', topn=5)
print("Mots contextuels pour 'Morocco':", context_words)
