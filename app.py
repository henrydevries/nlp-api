import platform
import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS
# import nltk
# import re
# import emoji
import contractions
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer, PorterStemmer
# from nltk import pos_tag
# from nltk.corpus import wordnet
import pandas as pd

# ////////////////////////////////////////////////////////////////////////
# TODO: Adjust the code part below
# ////////////////////////////////////////////////////////////////////////
# import nltk

# NLTK_DATA_PATH = "/app/nltk_data"

# # Zet de download directory expliciet
# nltk.data.path.append(NLTK_DATA_PATH)

# # Download alleen als niet aanwezig
# def safe_download(package):
#     try:
#         nltk.data.find(package)
#     except LookupError:
#         nltk.download(package.split("/")[-1], download_dir=NLTK_DATA_PATH)

# # Alleen eenmalig bij app-start
# safe_download("tokenizers/punkt")
# safe_download("corpora/stopwords")
# safe_download("corpora/wordnet")
# safe_download("taggers/averaged_perceptron_tagger")

# nltk.data.path.append(os.path.join(os.path.dirname(__file__), "bin", "nltk_data"))

# custom_punctuation = "!\"$%#&'()*+,-./:;=?@[\\]^_`{|}~”“…–’"

# emoticon_dict = {
#     ":)": "Positive",
#     ":-)": "Positive",
#     ";-)": "Positive",
#     ":D": "Positive",
#     ":d": "Positive",
#     "XD": "Positive",
#     "xd": "Positive",
#     "xD": "Positive",
#     "Xd": "Positive",
#     "<3": "Positive",
#     ";)": "Positive",
#     ":-P": "Positive",
#     ":-p": "Positive",
#     ":P": "Positive",
#     ":p": "Positive",
#     "\o/": "Positive",
#     "^_^": "Positive",
#     ":')": "Positive",
#     ":'D": "Positive",
#     "8-)": "Positive",
#     "8)": "Positive",
#     ":3": "Positive",
#     ":>": "Positive",
#     ":(": "Negative",
#     ":-(": "Negative",
#     "):": "Negative",
#     ":\\": "Negative",
#     ":/": "Negative",
#     ":'(": "Negative",
#     "-.-": "Negative",
#     "D:": "Negative",
#     ">:(": "Negative",
#     ">.<": "Negative",
#     ":-/": "Negative",
#     ":-\\": "Negative",
#     "=(" : "Negative",
#     "=/": "Negative",
#     "T_T": "Negative",
#     "D-:": "Negative",
#     ":-|": "Neutral", 
#     ":|": "Neutral",
#     "-_-": "Neutral",
#     "o_O": "Neutral",
#     "O_o": "Neutral",
#     ":o": "Neutral",
#     ":O": "Neutral",
#     ":-o": "Neutral",
#     ":-O": "Neutral",
#     "🤢": "Negative",
#     "🤬": "Negative",
#     "🤨": "Negative",
#     "🙃": "Neutral",
#     "🙄": "Negative",
#     "🙁": "Negative",
#     "🤮": "Negative",
#     "🤞": "Neutral",
#     "🥵": "Neutral",
#     "🙂": "Positive",
#     "🥺": "Positive",
#     "🤔": "Negative",
#     "🦄": "Neutral",
#     "🧼": "Neutral",
#     "🤣": "Positive",
#     "🖤": "Neutral",
#     "🤐": "Negative",
#     "🍿": "Neutral",
#     "🤙": "Positive",
#     "🥰": "Positive",
#     "🤪": "Positive",
#     "🤠": "Positive",
#     "🦈": "Neutral",
#     "🥴": "Negative",
#     "🤗": "Positive",
#     "🤤": "Positive",
#     "🤩": "Positive",
#     "🧤": "Neutral"
#     }

# def decode_unicode_escapes(text):
#     #Some observations contain \u002c or \u2019 etc. We need to decode these such that the processing to punctation goes correct when processing the data
#     if '\\u' in text :
#         try:
#             text = bytes(text, 'utf-8').decode('unicode_escape')
#         except Exception:
#             pass
#     return text

# def sentimentmaker_emoticons(df_emoji):
#     if df_emoji['Positive'] >= df_emoji['Negative'] and df_emoji['Positive'] >= df_emoji['Neutral']:
#         return 'Positive'
#     elif df_emoji['Negative'] >= df_emoji['Neutral']:
#         return 'Negative'
#     else:
#         return 'Neutral'

# def remove_color_gender(text):
#     #All skin tones for emojis need to be removed in order for the emoji dictionary to work properly. Addtionally the gender is also removed
#     skin_colors = ['\U0001F3FB', '\U0001F3FC', '\U0001F3FD', '\U0001F3FE', '\U0001F3FF']
#     genders = ['\u2640', '\u2642', '\uFE0F', '\u200D']

#     for tone in skin_colors:
#         text = text.replace(tone, "") 

#     for tone in genders:
#         text = text.replace(tone, "") 
#     return text

# def POS(tag):
#     if tag.startswith('J'):
#         return wordnet.ADJ
#     elif tag.startswith('V'):
#         return wordnet.VERB
#     elif tag.startswith('N'):
#         return wordnet.NOUN
#     elif tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return wordnet.NOUN

# def remove_stopwords(words):
#     #Define stop_words as downloaded stopwords
#     stop_words = set(stopwords.words('english'))
#     text_without_stopwords = []
#     for word in words:
#         if word not in stop_words:
#             text_without_stopwords.append(word)
#     text = ' '.join(text_without_stopwords)
#     return text

# def lemmatizer(words):
#     lemmatized_words = []
#     tagged_words = pos_tag(words)
    
#     for word, tag in tagged_words:
#         tagged = POS(tag)
#         lemma = WordNetLemmatizer().lemmatize(word, pos = tagged)
#         lemmatized_words.append(lemma)
#     text = ' '.join(lemmatized_words)
#     return text

# def stemmatizer(words):
#     stemmed_words = []
#     for word in words:
#         stem = PorterStemmer().stem(word)
#         stemmed_words.append(stem)  
#     text = ' '.join(stemmed_words)
#     return text

# def preprocessor(text, lowercase, no_adds, hashtags, no_url, no_punctuation, no_stopwords, stemming, lemmatization):


#     #Make from I'm --> I am
#     text = contractions.fix(text)

#     if lowercase:
#         text = text.lower()
#     if no_url:
#         text = re.sub(r'http\S+', '<url>', text)
    
#     #We need to remove color and gender in order for the dictionary to work properly
#     text = remove_color_gender(text)

#     #replace emoticons with sentiment words based on emoji sentiment score   
#     emoji_to_sentiment = dict(zip(df_emoji['Emoji'], df_emoji['Sentiment']))
#     for emoticon, replacement in emoji_to_sentiment.items():
#         text = text.replace(emoticon, ' ' + replacement + ' ')

#     #Replace emoticons from own dictionary
#     for emoticon, replacement in emoticon_dict.items():
#         text = text.replace(emoticon, ' ' + replacement + ' ')
    
#     #Convert not captured emojis to text
#     text = emoji.demojize(text, delimiters=("<", ">"))

#     #manually fixing two omportant emojis that are not being captured
#     text = text.replace("<person_shrugging>", "Neutral")
#     text = text.replace("<person_facepalming>", "Negative")

#     #After lowercasing for good formatting of the months
#     text = re.sub(
#     r'\b(?:\d{1,2}(?:st|nd|rd|th)?[\/\-. ](?:\d{1,2}|[A-Za-z]+)[\/\-. ]\d{2,4}'                                 #Day month year, with different seperators, so 5th March 2013 or 05/03/2013 for example
#     r'|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}'    #Month day year
#     r'|\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+\d{4}'        #Day month year
#     r'|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}'                                    #Month year
#     r'|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?'                  #Month day
#     r')\b', '<date>', text)
    
#     if no_adds:
#         text = re.sub(r'@\w+', '<user>', text)

#     if hashtags:
#         text = re.sub(r'#(\w+)', r'<hashtag> \1 <<hashtag>', text)

#     #Reduce elongated words to at most 2 repesitions of a character
#     text = re.sub(r'(.)\1{2,}', r'\1\1', text)

#     #Make of every number the expression <number> and times like 5am or 6pm to <time>
#     text = re.sub(r'\b\d{1,2}(:\d{2})?\s*(am|pm)\b', '<time>', text)
#     text = re.sub(r'\b\d+\s*[-/]\s*\d+\b', '<number>', text)
#     text = re.sub(r'\b\d+\b(?:\s+\b\d+\b)+', '<number>', text)
#     text = re.sub(r'\b\d+(?:st|nd|rd|th)?\b', '<number>', text)

#     if no_punctuation:
#         for char in custom_punctuation:
#             text = text.replace(char, ' ')
            
#     words = text.split()

#     if no_stopwords:
#         text = remove_stopwords(words)
#         words = text.split()

#     # Lemmatization with Part_Of_Speech tagging
#     if lemmatization and not stemming:
#         text = lemmatizer(words)

#     if stemming and not lemmatization:
#         text = stemmatizer(words)

#     return text

# df_emoji = pd.read_csv('Emoji_Sentiment_Data_v1.0.csv')
# df_emoji['Sentiment'] = df_emoji.apply(sentimentmaker_emoticons, axis=1)

GROUP_ID = 'the-chart-champions' # TODO: Replace with your groupID
MODEL_FILE = 'SVM.model' # relative path to your model file
MODEL_VERSION = 'v1.0'

# TODO: Adjust the function below so that it calls your vectorizer and 
# classifier functions packaged in the .model file.
def batch_predict(model, items):
    results = []
    # options = model['preprocessing_options']


    for item in items:
        # cleaned_text = preprocessor(decode_unicode_escapes(item['text']), **model["preprocessing_options"])
        X = model['vectorizer'].transform([cleaned_text]) #was fit_transform
        X = model['selector'].transform(X)
        label = model['classifier'].predict(X)
        results.append({
            "id": item['id'],
            "label": int(label[0] - 1),
        })
    return results


# ////////////////////////////////////////////////////////////////////////
# You should not(
# ////////////////////////////////////////////////////////////////////////

app = Flask(__name__) # set up app
CORS(app) # set up CORS policies

# load model file
with open(MODEL_FILE, 'rb') as file:
    model = pickle.load(file)

# define meta-data for API
meta_data = {
    "groupID": GROUP_ID,
    "modelFile": MODEL_FILE,
    "modelVersion": MODEL_VERSION,
    "pythonVersion": platform.python_version()
}

# api route
@app.route("/", methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        items = request.json['items']
        return jsonify({ "items": batch_predict(model, items) }) # batch predictions
    else:
        return jsonify({"meta": meta_data}) # meta data

# start the api server when running the script
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)