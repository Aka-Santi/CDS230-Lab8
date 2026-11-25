# Naive Bayes Lab - Clickbait detector
# Santiago Rodriguez Gomez
#
# same thing as example 1 (spam filter) but with headlines instead of emails
# dataset: https://www.kaggle.com/datasets/amananandrai/clickbait-dataset
#   headline = the text
#   clickbait = 1 if clickbait, 0 if normal news
#
# P(clickbait | words) = P(words | clickbait) * P(clickbait) / P(words)

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer   # words -> counts
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load the data (example 1 typed 10 emails by hand, this has 32k headlines)
df = pd.read_csv("clickbait_data.csv")

print("=== STEP 2: Our Dataset ===")
print(df.head())
print()
print(f"Total headlines     : {len(df)}")
print(f"Clickbait headlines : {sum(df['clickbait'] == 1)}")
print(f"Normal headlines    : {sum(df['clickbait'] == 0)}")
print()

headlines = df["headline"]   # basically the "emails" list from ex 1
labels    = df["clickbait"]  # and the "labels" list

# split 80/20 so we can test on headlines it hasnt seen
# ex 1 didnt do this bc it only had 10 emails
# stratify keeps it 50/50 in both parts
train_text, test_text, train_labels, test_labels = train_test_split(
    headlines, labels,
    test_size=0.2,
    random_state=42,   # so i get the same split every run
    stratify=labels
)

print("=== STEP 3: Train/Test split ===")
print(f"Training headlines: {len(train_text)}")
print(f"Testing headlines : {len(test_text)}")
print()

# turn words into counts, same as ex 1
# fit_transform ONLY on train, then just transform on test
# (otherwise the test words leak into the vocab)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

print("=== STEP 4: Words converted to numbers ===")
print(f"Number of unique words found: {len(vectorizer.get_feature_names_out())}")
# full list is like 20k words so just print a few
print(f"Sample of words: {vectorizer.get_feature_names_out()[1000:1015]}")
print()

# train
model = MultinomialNB()
model.fit(X_train, train_labels)

print("=== STEP 5: Model trained! ===")
print()

# check accuracy on the test set
test_predictions = model.predict(X_test)

print("=== STEP 6: How accurate is the model? ===")
print(f"Accuracy: {accuracy_score(test_labels, test_predictions) * 100:.1f}%")
print()
print("Confusion matrix (rows = real answer, columns = prediction):")
print("              Pred Normal  Pred Clickbait")
cm = confusion_matrix(test_labels, test_predictions)
print(f"Real Normal    {cm[0][0]:>10}  {cm[0][1]:>14}")
print(f"Real Clickbait {cm[1][0]:>10}  {cm[1][1]:>14}")
print()
print(classification_report(test_labels, test_predictions,
                            target_names=["Normal", "Clickbait"]))

# which words are the most clickbait-y?
# feature_log_prob_ has log P(word | class) for each class
# so clickbait - normal = how much more a word leans clickbait
words = vectorizer.get_feature_names_out()
clickbait_score = model.feature_log_prob_[1] - model.feature_log_prob_[0]

top_clickbait = clickbait_score.argsort()[-15:][::-1]   # highest
top_normal    = clickbait_score.argsort()[:15]          # lowest

print("=== STEP 7: Most telling words ===")
print(f"Top clickbait words: {[words[i] for i in top_clickbait]}")
print(f"Top normal words   : {[words[i] for i in top_normal]}")
print()

# test on some headlines i made up
new_headlines = [
    "You Won't Believe What This Dog Did Next",              # clickbait
    "Senate passes budget bill after long debate",           # normal
    "17 Things Only College Students Will Understand",       # clickbait
    "Earthquake strikes off the coast of Japan",             # normal
    "This Naive Bayes Lab Will Change The Way You See Data", # lol
]

X_new = vectorizer.transform(new_headlines)   # transform only, same vocab
predictions   = model.predict(X_new)
probabilities = model.predict_proba(X_new)

print("=== STEP 8: Predictions on new headlines ===")
print()

for i, headline in enumerate(new_headlines):
    result = "CLICKBAIT" if predictions[i] == 1 else "NORMAL"

    # [0] = normal, [1] = clickbait
    prob_clickbait = probabilities[i][1] * 100
    prob_normal    = probabilities[i][0] * 100

    print(f"Headline: '{headline}'")
    print(f"  --> Prediction      : {result}")
    print(f"  --> Clickbait chance: {prob_clickbait:.1f}%")
    print(f"  --> Normal chance   : {prob_normal:.1f}%")
    print()

# ============================================================
# EXPLANATION (for a general audience)
# ============================================================
# So what is this actually doing?
#
# You know those headlines like "You Wont Believe What Happened Next"
# or "17 Things Only 90s Kids Will Get"? Thats clickbait. This program
# reads a headline and guesses if its clickbait or a normal news headline,
# kinda like how a spam filter guesses if an email is spam.
#
# How does it "learn"? First it turns every headline into word counts,
# since computers cant read words, they only understand numbers. Then
# Naive Bayes just counts how often each word shows up in clickbait vs
# normal headlines. For example "you" and "things" show up way more in
# clickbait, while words like "kills", "leader" and "crash" show up way
# more in real news. When a new headline comes in, it checks every word
# and adds up the evidence for each side, then picks the winner.
#
# The "naive" part means it looks at each word on its own and ignores
# word order. So to the model "dog bites man" and "man bites dog" are
# the exact same headline. Not perfect but it works suprisingly well.
#
# How did it do?
#   - I trained it on 25,600 headlines and tested it on 6,400 it never saw.
#   - It got 97% of them right, which is really good for something this
#     simple.
#   - Most of its mistakes were normal headlines getting flagged as
#     clickbait (152), only 43 clickbait ones slipped through.
#
# Something interesting tho: some of the top "clickbait words" are
# "2015", "2016" and "instagram". Thats not because years are clickbait,
# its because most of the clickbait in this dataset came from BuzzFeed
# around those years. So the model is partly learning WHERE and WHEN the
# headline came from, not just how its written. If you gave it clickbait
# from 2026 it might not do as good.
#
# Also the 100% / 0% chances are not really "100% sure". Naive Bayes
# multiplies a ton of small numbers together so it gets overconfident
# really fast, especially with longer headlines.
# ============================================================
