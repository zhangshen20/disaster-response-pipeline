import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')


def tokenize(text):
    """Tokenize text. Remove stop words. Lemmatize and stem words
    
    Args:
        text: str. 
    
    Return:
        A list containing processed text
    """
    words = word_tokenize(text.lower().strip())
    
    # Remove Stop Words
    words = [w for w in words if w not in stopwords.words("english")]
    
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    
    stemmed = [PorterStemmer().stem(w) for w in lemmed]
    
    return stemmed
