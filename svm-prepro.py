import sqlite3
import optuna
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.feature_selection import SelectKBest, chi2, SelectFpr
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC
import pickle


def startoclass(star):
    if star <= 2:
        return -1
    elif star == 3:
        return 0
    elif star >= 4:
        return 1

def sentimentmaker(df):
    df = df.copy()
    df['sentiment'] = df['label']
    reviews = df['type'] == "Review"
    df.loc[reviews, 'sentiment'] = df.loc[reviews, 'label'].apply(startoclass)
    return df

def dataloader(source, total_data):
    '''
    source:
    - 'Tweet', getting only tweets
    - 'Review', getting only reviews
    - 'all', getting all values

    total_data:
    - True, getting total dataset
    - False, getting random 1000 observations

    output: clean dataset consisting of label, text and sentiment
    '''
    DB_FILE = '/local/DSPT/data/nlp-data.db'
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    base_query = '''
    SELECT Labels.NumericValue, RawTexts.Text, Documents.Type
    FROM Documents
    INNER JOIN RawTexts ON Documents.RawTextID = RawTexts.RawTextID
    INNER JOIN Labels ON Documents.LabelID = Labels.LabelID
    '''

    if source == "Tweet":
        base_query += 'WHERE Documents.Type = "Tweet"\n'
    elif source == "Review":
        base_query += 'WHERE Documents.Type = "Review"\n'
    elif source != "all":
        raise ValueError("Invalid source")

    if not total_data:
        base_query += 'ORDER BY RANDOM() LIMIT 10000'

    query = cur.execute(base_query)
    data = pd.DataFrame(query.fetchall(), columns=['label', 'text', 'type'])
    con.close()
    data = sentimentmaker(data)

    #removing all conflicting tweets
    grouped = data.groupby('text')['sentiment'].nunique()
    conflicted_texts = grouped[grouped > 1].index
    data_cleaned = data[~((data['type'] == 'Tweet') & (data['text'].isin(conflicted_texts)))]
    print(data_cleaned.head())
    return data_cleaned

def samplesplit(X,y,val=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23, shuffle=True)
    if val:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=23, stratify=y)
        return X_train, y_train, X_val, y_val, X_test, y_test
    return X_train, y_train, X_test, y_test

def make_objective(X, y):
    f1_macro_scorer = make_scorer(f1_score, average='macro', zero_division=0)
    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 0.01, 1),
            'class_weight': 'balanced'
        }

        model = LinearSVC(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=23)
        scores = cross_val_score(model, X, y, cv=cv, scoring=f1_macro_scorer)
        return np.mean(scores)
    
    return objective

#Preprocessor for all data
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import emoji
import contractions

#nltk.download('all')
    
custom_punctuation = "!\"$%#&'()*+,-./:;=?@[\\]^_`{|}~”“…–’"

emoticon_dict = {
    ":)": "Positive",
    ":-)": "Positive",
    ";-)": "Positive",
    ":D": "Positive",
    ":d": "Positive",
    "XD": "Positive",
    "xd": "Positive",
    "xD": "Positive",
    "Xd": "Positive",
    "<3": "Positive",
    ";)": "Positive",
    ":-P": "Positive",
    ":-p": "Positive",
    ":P": "Positive",
    ":p": "Positive",
    "\o/": "Positive",
    "^_^": "Positive",
    ":')": "Positive",
    ":'D": "Positive",
    "8-)": "Positive",
    "8)": "Positive",
    ":3": "Positive",
    ":>": "Positive",
    ":(": "Negative",
    ":-(": "Negative",
    "):": "Negative",
    ":\\": "Negative",
    ":/": "Negative",
    ":'(": "Negative",
    "-.-": "Negative",
    "D:": "Negative",
    ">:(": "Negative",
    ">.<": "Negative",
    ":-/": "Negative",
    ":-\\": "Negative",
    "=(" : "Negative",
    "=/": "Negative",
    "T_T": "Negative",
    "D-:": "Negative",
    ":-|": "Neutral", 
    ":|": "Neutral",
    "-_-": "Neutral",
    "o_O": "Neutral",
    "O_o": "Neutral",
    ":o": "Neutral",
    ":O": "Neutral",
    ":-o": "Neutral",
    ":-O": "Neutral",
    "🤢": "Negative",
    "🤬": "Negative",
    "🤨": "Negative",
    "🙃": "Neutral",
    "🙄": "Negative",
    "🙁": "Negative",
    "🤮": "Negative",
    "🤞": "Neutral",
    "🥵": "Neutral",
    "🙂": "Positive",
    "🥺": "Positive",
    "🤔": "Negative",
    "🦄": "Neutral",
    "🧼": "Neutral",
    "🤣": "Positive",
    "🖤": "Neutral",
    "🤐": "Negative",
    "🍿": "Neutral",
    "🤙": "Positive",
    "🥰": "Positive",
    "🤪": "Positive",
    "🤠": "Positive",
    "🦈": "Neutral",
    "🥴": "Negative",
    "🤗": "Positive",
    "🤤": "Positive",
    "🤩": "Positive",
    "🧤": "Neutral"
    }

def decode_unicode_escapes(text):
    #Some observations contain \u002c or \u2019 etc. We need to decode these such that the processing to punctation goes correct when processing the data
    if '\\u' in text :
        try:
            text = bytes(text, 'utf-8').decode('unicode_escape')
        except Exception:
            pass
    return text

def sentimentmaker_emoticons(df_emoji):
    if df_emoji['Positive'] >= df_emoji['Negative'] and df_emoji['Positive'] >= df_emoji['Neutral']:
        return 'Positive'
    elif df_emoji['Negative'] >= df_emoji['Neutral']:
        return 'Negative'
    else:
        return 'Neutral'

def remove_color_gender(text):
    #All skin tones for emojis need to be removed in order for the emoji dictionary to work properly. Addtionally the gender is also removed
    skin_colors = ['\U0001F3FB', '\U0001F3FC', '\U0001F3FD', '\U0001F3FE', '\U0001F3FF']
    genders = ['\u2640', '\u2642', '\uFE0F', '\u200D']

    for tone in skin_colors:
        text = text.replace(tone, "") 

    for tone in genders:
        text = text.replace(tone, "") 
    return text

def POS(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def remove_stopwords(words):
    #Define stop_words as downloaded stopwords
    stop_words = set(stopwords.words('english'))
    text_without_stopwords = []
    for word in words:
        if word not in stop_words:
            text_without_stopwords.append(word)
    text = ' '.join(text_without_stopwords)
    return text

def lemmatizer(words):
    lemmatized_words = []
    tagged_words = pos_tag(words)
    
    for word, tag in tagged_words:
        tagged = POS(tag)
        lemma = WordNetLemmatizer().lemmatize(word, pos = tagged)
        lemmatized_words.append(lemma)
    text = ' '.join(lemmatized_words)
    return text

def stemmatizer(words):
    stemmed_words = []
    for word in words:
        stem = PorterStemmer().stem(word)
        stemmed_words.append(stem)  
    text = ' '.join(stemmed_words)
    return text

def preprocessor(text, lowercase, no_adds, hashtags, no_url, no_punctuation, no_stopwords, stemming, lemmatization):


    #Make from I'm --> I am
    text = contractions.fix(text)

    if lowercase:
        text = text.lower()
    if no_url:
        text = re.sub(r'http\S+', '<url>', text)
    
    #We need to remove color and gender in order for the dictionary to work properly
    text = remove_color_gender(text)

    #replace emoticons with sentiment words based on emoji sentiment score   
    emoji_to_sentiment = dict(zip(df_emoji['Emoji'], df_emoji['Sentiment']))
    for emoticon, replacement in emoji_to_sentiment.items():
        text = text.replace(emoticon, ' ' + replacement + ' ')

    #Replace emoticons from own dictionary
    for emoticon, replacement in emoticon_dict.items():
        text = text.replace(emoticon, ' ' + replacement + ' ')
    
    #Convert not captured emojis to text
    text = emoji.demojize(text, delimiters=("<", ">"))

    #manually fixing two omportant emojis that are not being captured
    text = text.replace("<person_shrugging>", "Neutral")
    text = text.replace("<person_facepalming>", "Negative")

    #After lowercasing for good formatting of the months
    text = re.sub(
    r'\b(?:\d{1,2}(?:st|nd|rd|th)?[\/\-. ](?:\d{1,2}|[A-Za-z]+)[\/\-. ]\d{2,4}'                                 #Day month year, with different seperators, so 5th March 2013 or 05/03/2013 for example
    r'|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}'    #Month day year
    r'|\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+\d{4}'        #Day month year
    r'|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}'                                    #Month year
    r'|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?'                  #Month day
    r')\b', '<date>', text)
    
    if no_adds:
        text = re.sub(r'@\w+', '<user>', text)

    if hashtags:
        text = re.sub(r'#(\w+)', r'<hashtag> \1 <<hashtag>', text)

    #Reduce elongated words to at most 2 repesitions of a character
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    #Make of every number the expression <number> and times like 5am or 6pm to <time>
    text = re.sub(r'\b\d{1,2}(:\d{2})?\s*(am|pm)\b', '<time>', text)
    text = re.sub(r'\b\d+\s*[-/]\s*\d+\b', '<number>', text)
    text = re.sub(r'\b\d+\b(?:\s+\b\d+\b)+', '<number>', text)
    text = re.sub(r'\b\d+(?:st|nd|rd|th)?\b', '<number>', text)

    if no_punctuation:
        for char in custom_punctuation:
            text = text.replace(char, ' ')
            
    words = text.split()

    if no_stopwords:
        text = remove_stopwords(words)
        words = text.split()

    # Lemmatization with Part_Of_Speech tagging
    if lemmatization and not stemming:
        text = lemmatizer(words)

    if stemming and not lemmatization:
        text = stemmatizer(words)

    return text

#Do not set 'stemming' and 'lemmatization' both to True. This will result in no stemming or lemmatization at all.
#Note that for stopwords and lemmatization to work properly, lowercase must be True.

df_emoji = pd.read_csv('Emoji_Sentiment_Data_v1.0.csv')
df_emoji['Sentiment'] = df_emoji.apply(sentimentmaker_emoticons, axis=1)

data = dataloader("all", True)
data['text'] = data['text'].apply(decode_unicode_escapes)

data['cleaned_text'] = data['text'].apply(lambda x: preprocessor(x, 
            lowercase=True, 
            no_adds=True, 
            hashtags=True, 
            no_url=True, 
            no_punctuation=True,
            no_stopwords=False,
            lemmatization=True,
            stemming=False))

print("Preproccesing is done")

X_train, y_train, X_test, y_test = samplesplit(data['cleaned_text'].values, data['sentiment'].values)
y_train, y_test = y_train +1, y_test +1 
    
vectorizer = TfidfVectorizer(ngram_range = (1,2))
X_train = vectorizer.fit_transform(X_train)

from sklearn.utils.validation import check_is_fitted

try:
    check_is_fitted(vectorizer, 'vocabulary_')
    print("Vectorizer is fitted.")
except Exception as e:
    print("Vectorizer is NOT fitted.")
    print(e)

X_test = vectorizer.transform(X_test)
vocabulary = vectorizer.vocabulary_
print(X_train.shape)
    
selector = SelectFpr(score_func = chi2, alpha = 0.15)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)
print(X_train.shape)
    
objective = make_objective(X_train, y_train)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=3600*3)
best_params = study.best_params
print(best_params)
    
model = LinearSVC(**best_params)
    
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_insample = model.predict(X_train)
    
print("########## OUT OF SAMPLE ##########")

acc = accuracy_score(y_test, y_pred)
prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {prec_macro:.4f}")
print(f"Recall (macro): {rec_macro:.4f}")
print(f"F1-score (macro): {f1_macro:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("########## IN SAMPLE ##########")
acc = accuracy_score(y_train, y_pred_insample)
prec_macro = precision_score(y_train, y_pred_insample, average='macro', zero_division=0)
rec_macro = recall_score(y_train, y_pred_insample, average='macro', zero_division=0)
f1_macro = f1_score(y_train, y_pred_insample, average='macro', zero_division=0)

print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {prec_macro:.4f}")
print(f"Recall (macro): {rec_macro:.4f}")
print(f"F1-score (macro): {f1_macro:.4f}")

print("\nClassification Report:")
print(classification_report(y_train, y_pred_insample, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_insample))

model_pipeline = {
    "vectorizer": vectorizer,
    "selector": selector,
    "classifier": model,
    "preprocessing_options": {
        "lowercase": True,
        "no_adds": True,
        "hashtags": True,
        "no_url": True,
        "no_punctuation": True,
        "no_stopwords": False,
        "lemmatization": True,
        "stemming": False }
}


with open("SVM.model", "wb") as f:
    pickle.dump(model_pipeline, f)