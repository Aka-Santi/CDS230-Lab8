# Naive Bayes Lab - is this masked hero sighting real?
# Santiago Rodriguez Gomez
#
# same thing as example 2 (tennis) but with ~86k fake hero sightings
# dataset: Masked Hero Tracker, sightings.csv (all synthetic, no real ppl)
#
# features (all categories, like outlook/humidity/wind in ex 2):
#   report_type, borough, weather_condition, report_source,
#   movement_direction, and a bunch of True/False ones
#   (photo/video/audio evidence, web residue, police report, crime nearby)
#
# label: is_verified -> 1 = real, 0 = fake or mistake

import pandas as pd
from sklearn.naive_bayes import CategoricalNB   # NB for category data
from sklearn.preprocessing import LabelEncoder  # words -> numbers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load data (ex 2 was 14 rows typed by hand, this is a csv w/ 86k)
df = pd.read_csv("sightings.csv")

feature_columns = [
    "report_type",
    "borough",
    "weather_condition",
    "report_source",
    "movement_direction",
    "photo_evidence",
    "video_evidence",
    "audio_evidence",
    "web_residue_found",
    "police_report_linked",
    "crime_nearby",
]

# NOT using verification_status bc thats basically the answer
# and tracker_confidence is a number not a category

print("=== STEP 2: Our Dataset ===")
print(df[feature_columns + ["is_verified"]].head())   # just first 5, its huge
print()
print(f"Total sightings   : {len(df)}")
print(f"Real sightings    : {sum(df['is_verified'] == 1)}")
print(f"Fake/mistaken ones: {sum(df['is_verified'] == 0)}")
print()

# words -> numbers
# ex 2 made 4 encoders by hand, i have 11 cols so just loop
# saving each encoder in a dict so i can reuse it for new sightings later
encoders = {}

for column in feature_columns:
    encoder = LabelEncoder()
    df[column + "_num"] = encoder.fit_transform(df[column])
    encoders[column] = encoder

print("=== STEP 3: Example of words converted to numbers ===")
for column in ["report_type", "report_source", "photo_evidence"]:
    categories = encoders[column].classes_
    mapping = {str(cat): i for i, cat in enumerate(categories)}
    print(f"{column}: {mapping}")
print()

# split into train/test
# the dataset already has a "split" column so just use that
numeric_columns = [column + "_num" for column in feature_columns]

train = df[df["split"] == "train"]
test  = df[df["split"] == "test"]

X_train = train[numeric_columns]
y_train = train["is_verified"]
X_test  = test[numeric_columns]
y_test  = test["is_verified"]

print("=== STEP 4: Train/Test split ===")
print(f"Training sightings: {len(train)}")
print(f"Testing sightings : {len(test)}")
print()

# train
model = CategoricalNB()
model.fit(X_train, y_train)

print("=== STEP 5: Model trained! ===")
print()

# check accuracy on test set
test_predictions = model.predict(X_test)

# baseline = just guessing "real" every time
# if the model cant beat this its useless
always_real_accuracy = (y_test == 1).mean()

print("=== STEP 6: How accurate is the model? ===")
print(f"Model accuracy            : {accuracy_score(y_test, test_predictions) * 100:.1f}%")
print(f"Always-guess-REAL accuracy: {always_real_accuracy * 100:.1f}%")
print()
print("Confusion matrix (rows = real answer, columns = prediction):")
print("               Pred Fake  Pred Real")
cm = confusion_matrix(y_test, test_predictions)
print(f"Actually Fake  {cm[0][0]:>9}  {cm[0][1]:>9}")
print(f"Actually Real  {cm[1][0]:>9}  {cm[1][1]:>9}")
print()
print(classification_report(y_test, test_predictions, target_names=["Fake", "Real"]))

# % real by report source
# NB is just counting anyway so this is kinda what its looking at
print("=== STEP 7: % of sightings that were real, by report source ===")
source_rates = train.groupby("report_source")["is_verified"].mean().sort_values(ascending=False)
for source, rate in source_rates.items():
    print(f"  {source:<15} {rate * 100:5.1f}% real")
print()

# predict some new sightings i made up
# have to use transform (not fit_transform) w/ the saved encoders
new_sightings = [
    # tons of evidence + police + crime nearby, should be real
    {"report_type": "crime_response", "borough": "Manhattan", "weather_condition": "clear",
     "report_source": "police_scanner", "movement_direction": "N",
     "photo_evidence": True, "video_evidence": True, "audio_evidence": False,
     "web_residue_found": True, "police_report_linked": True, "crime_nearby": True},

    # random social media post in the fog w/ no evidence, prob fake
    {"report_type": "swinging", "borough": "Staten_Island", "weather_condition": "fog",
     "report_source": "social_media", "movement_direction": "SW",
     "photo_evidence": False, "video_evidence": False, "audio_evidence": False,
     "web_residue_found": False, "police_report_linked": False, "crime_nearby": False},

    # somewhere in the middle
    {"report_type": "rooftop", "borough": "Brooklyn", "weather_condition": "rain",
     "report_source": "mobile_app", "movement_direction": "E",
     "photo_evidence": True, "video_evidence": False, "audio_evidence": False,
     "web_residue_found": False, "police_report_linked": False, "crime_nearby": False},
]

print("=== STEP 8: Predictions for new sightings ===")
print()

for sighting in new_sightings:
    # encode each value w/ its column's encoder
    single_input = [[encoders[column].transform([sighting[column]])[0]
                     for column in feature_columns]]
    single_input = pd.DataFrame(single_input, columns=numeric_columns)

    prediction  = model.predict(single_input)[0]
    probability = model.predict_proba(single_input)[0]

    result = "REAL" if prediction == 1 else "FAKE"

    print(f"Sighting: {sighting['report_type']} in {sighting['borough']}, "
          f"{sighting['weather_condition']}, via {sighting['report_source']}")
    print(f"  --> Is it real? {result}")
    print(f"  --> Probability REAL: {probability[1] * 100:.1f}%")
    print(f"  --> Probability FAKE: {probability[0] * 100:.1f}%")
    print()

# ============================================================
# EXPLANATION (for a general audience)
# ============================================================
# So what is this actually doing?
#
# Imagine you run a hotline where people call in saying they saw a
# masked hero swinging around the city. Some of those calls are real,
# some are people who saw a guy in a costume, and some are straight up
# hoaxes. This program tries to guess which calls are real just from
# basic info about the report, like where it happened, what the weather
# was, how it got reported, and if there was a photo or video.
#
# How does it "learn"? Naive Bayes is basically just counting. It looks
# at the 69,000 training sightings and keeps track of stuff like "out of
# all the real sightings, how many came from a police scanner?" and "out
# of all the fake ones, how many came from social media?". Then when a
# new sighting comes in it multiplies those odds together for every
# clue and picks whichever answer (real or fake) comes out higher.
#
# The "naive" part means it treats every clue like its totally separate
# from the others. In real life thats not true (if theres a video theres
# probably a photo too) but it still works pretty well.
#
# How did it do?
#   - It got about 73% of the test sightings right.
#   - If you just said "real" every single time you would get 63%, so the
#     model is actually learning something, its not just guessing.
#   - Its better at spotting real ones (85% caught) than fake ones (only
#     about half caught). A lot of fakes slip through because the dataset
#     was made so some fakes, like impersonators, look almost the same as
#     real sightings on paper.
#
# The coolest thing I found is how much the report source matters.
# Police scanner reports were real like 92% of the time, while social
# media reports were real only about 30% of the time. So basically dont
# trust the internet lol.
#
# One thing to keep in mind is that the 100% / 0% probabilities on the
# first new sighting are not really "100% sure". Naive Bayes multiplies
# a lot of numbers together so it tends to be overconfident when a bunch
# of clues all point the same way.
#
# Also this whole dataset is fake (made by a simulation), so its just for
# practice, no real people or real crimes are in here.
# ============================================================
