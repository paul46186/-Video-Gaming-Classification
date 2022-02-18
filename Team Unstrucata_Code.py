import bs4
import pandas as pd
import requests
import time
import random

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36"
}

cities = ['newyork', 'chicago', 'miami', 'philadelphia', 'washingtondc', 'losangeles']

df = pd.DataFrame(columns=['link', 'post_id', 'post_date', 'city', 'title', 'price', 'place', 'desc'])

## Timeout error encountered sometimes when run as a loop. To retrieve the data, we scraped each city individually. 
## Additionally, this code is time consuming to run as a loop.

for city in cities:
    print(city)
    base_url = 'https://' + city + '.craigslist.org/search/vga'
    re = requests.get(base_url, headers=headers)
    
    soup = bs4.BeautifulSoup(re.text)
    
    # find the total number of pages for the city
    count = int(soup.select('.totalcount')[0].getText())
    num_pages = count // 120
    
    for page in range(num_pages):
        base_url = 'https://' + city + '.craigslist.org/search/vga?s=' + str(page*120)
        re = requests.get(base_url, headers=headers)
        soup = bs4.BeautifulSoup(re.text)
        
        # only use HTML tags of tags that have the 'result-image' tag
        soup = soup.select('.result-image')
        
        # create a list of all the links on the page
        links = [x.attrs['href'] for x in soup]
        
        # loop through each listing on this page
        for link in links:
            posting_re = requests.get(link)
            posting_soup = bs4.BeautifulSoup(posting_re.text)
            
            # pretty straightforward
            try:
                title = posting_soup.select('#titletextonly')[0].getText()
            except:
                continue
                
            try:
                price = posting_soup.select('.price')[0].getText()
            except:
                continue
                
            try:
                place = posting_soup.select('small')[0].getText()
            except:
                continue
                
            try:
                desc = posting_soup.select('section[id="postingbody"]')[0].getText()
            except:
                continue
            
            try:
                post_id = posting_soup.select('p[class="postinginfo"]')[0].getText()[9:]
            except:
                continue
                
            try:
                post_date = posting_soup.select('time')[1].getText()
            except:
                continue
            
            # add this listing to our DataFrame
            df = df.append([[link, post_id, post_date, city, title, price, place, desc]], ignore_index=True)
            
            time.sleep(random.randint(0,3))


# # Data Cleaning
allclean = df

# ### Cleaning Description Column
descrip = []
for d in allclean["desc"]:
    des = ""
    for s in d.split("\n\n\n")[1].split("\n"):
        des += s
    descrip.append(des.lower())
allclean["desc"] = descrip


# ### Cleaning Title Column
allclean["title"] = allclean["title"].str.lower()


### Creating our Target Variable
brands = {"Microsoft": ["xbox", "microsoft", "360", "xbox one", " rig ", "x box", "series x", "series s", "halo","kinect"], "Nintendo": ["wii", "gamecube", "game cube", " ds ", "gameboy", "nintendo", "switch", "cube", "lite", " dsi ", "2ds", "wii fit", "nunchuck", "game boy", "game boy", "onyx", "oled", "3ds", "gba","wii remote"], "Sony": ["playstation", "play station", " ps ", "sony", " psp ", "ps4", "ps3", "ps5", "ps vr", "pro", "ps4 slim"], "Arcade": ["arcade", "atari", "slot machine", "pinball", "machine"], "Meta": ["oculus", "meta", "quest", "quest 2", "vr headset", "rift", "quest 2 elite", "touch controller", "halo headband", "head strap"]}

import numpy as np
for brand in brands:
    allclean[brand] = np.repeat(0, 3011)
    for word in brands[brand]:
        allclean[brand] += allclean.title.str.contains(word)
        allclean[brand] += allclean.desc.str.contains(word)

allclean["Other"] = np.repeat(0.5, 3011)

allclean["Brand"] = allclean.iloc[:,-6:].idxmax(1)

# # Model Building
# ### Adding a Review Column
data = allclean
data["review"] = data["title"] + data["desc"]
data["review"] = data["review"].fillna("")

# ### Splitting the Data
from sklearn.model_selection import train_test_split
training_x, testing_x, train_y, test_y = train_test_split(data["review"], data["Brand"], test_size = 0.25, random_state = 12)


# ### Creating a Vocabulary and Vectorizing
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

lemmatizer = nltk.stem.WordNetLemmatizer()
tokencomp = []
for review in list(training_x):
    tokens = nltk.word_tokenize(str(review).lower())
    lemmatized_token = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    tokencomp.append([token for token in lemmatized_token if token not in stopwords.words('english')])

comp = []
for review in tokencomp:
    comp.append(" ".join(review))
vectorizer = TfidfVectorizer(ngram_range = (1,2), min_df = 2)
vectorizer.fit(comp)

train_x = vectorizer.transform(training_x)
test_x = vectorizer.transform(testing_x)


# ## Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
NBmodel = MultinomialNB()

NBmodel.fit(train_x, train_y)
y_pred_NB = NBmodel.predict(test_x)

acc_NB = accuracy_score(test_y, y_pred_NB)
print("Naive Bayes model Accuracy:: {:.2f}%".format(acc_NB*100))


# ## Logistic Model
from sklearn.linear_model import LogisticRegression
Logitmodel = LogisticRegression()

Logitmodel.fit(train_x, train_y)
y_pred_logit = Logitmodel.predict(test_x)

acc_logit = accuracy_score(test_y, y_pred_logit)
print("Logit model Accuracy:: {:.2f}%".format(acc_logit*100))


# ## Random Forest
from sklearn.ensemble import RandomForestClassifier

RFmodel = RandomForestClassifier(n_estimators=50, max_depth=6, bootstrap=True, random_state=0)

RFmodel.fit(train_x, train_y)
y_pred_RF = RFmodel.predict(test_x)

acc_RF = accuracy_score(test_y, y_pred_RF)
print("Random Forest Model Accuracy: {:.2f}%".format(acc_RF*100))


# ## Support Vector Classifier
from sklearn.svm import LinearSVC
SVMmodel = LinearSVC()

SVMmodel.fit(train_x, train_y)
y_pred_SVM = SVMmodel.predict(test_x)

acc_SVM = accuracy_score(test_y, y_pred_SVM)
print("SVM model Accuracy: {:.2f}%".format(acc_SVM*100))


# ## Neural Network
from sklearn.neural_network import MLPClassifier
DLmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(3,2), random_state=1)

DLmodel.fit(train_x, train_y)
y_pred_DL= DLmodel.predict(test_x)

acc_DL = accuracy_score(test_y, y_pred_DL)
print("DL model Accuracy: {:.2f}%".format(acc_DL*100))


# ## Recurrent Neural Network
docs_x = []
docs_train_x = []
docs_test_x = []
for review in training_x:
    docs_x.append(nltk.word_tokenize(str(review).lower()))
    docs_train_x.append(nltk.word_tokenize(str(review).lower()))
for review in testing_x:
    docs_x.append(nltk.word_tokenize(str(review).lower()))
    docs_test_x.append(nltk.word_tokenize(str(review).lower()))

from collections import Counter
words = [j for i in docs_x for j in i]
count_words = Counter(words)
total_words = len(words)
sorted_words = count_words.most_common(total_words)
vocab_to_int = {w: i+1 for i, (w,c) in enumerate(sorted_words)} 

text_int = []
for i in docs_train_x:
    r = [vocab_to_int[w] for w in i]
    text_int.append(r)


text_test_int = []
for i in docs_test_x:
    r = [vocab_to_int[w] for w in i]
    text_test_int.append(r)


from keras.preprocessing import sequence 
from keras.models import Sequential 
from keras.layers import Dense, Embedding, Flatten 
from keras.layers import LSTM
max_features = total_words
maxlen = 250
batch_size = 32

x_train = sequence.pad_sequences(text_int, maxlen=maxlen)
x_test = sequence.pad_sequences(text_test_int, maxlen=maxlen)

encoded_train = [0 if label =='Sony' else 1 if label == "Nintendo" else 2 if label == "Microsoft" else 3 if label == "Arcade" else 4 if label == "Meta" else 5 for label in train_y]
encoded_test = [0 if label =='Sony' else 1 if label == "Nintendo" else 2 if label == "Microsoft" else 3 if label == "Arcade" else 4 if label == "Meta" else 5 for label in test_y]

model = Sequential()
model.add(Embedding(max_features, 20, input_length=maxlen))
model.add(LSTM(100, dropout=0.10, recurrent_dropout=0.10))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train.tolist(), encoded_train, batch_size=batch_size, epochs=2, validation_data=(x_test.tolist(), encoded_test))


# ## Confusion Matrix for SVC Model
from sklearn.metrics import confusion_matrix
import numpy as np

pd.DataFrame(confusion_matrix(test_y, y_pred_SVM), index = ["Arcade", "Meta", "Microsoft", "Nintendo", "Other", "Sony"], columns = ["Arcade", "Meta", "Microsoft", "Nintendo", "Other", "Sony"])