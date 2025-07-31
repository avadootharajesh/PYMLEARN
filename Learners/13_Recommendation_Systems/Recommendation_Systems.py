# Recommendation_Systems.py
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Load built-in Movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# Train/test split
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Use user-based collaborative filtering
sim_options = {'name': 'cosine', 'user_based': True}
algo = KNNBasic(sim_options=sim_options)

# Train
algo.fit(trainset)

# Predict on test set
predictions = algo.test(testset)

# Evaluate RMSE
rmse(predictions)

# Predict rating for user 196 on item 302
uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid)
print(f"Predicted rating of user {uid} for item {iid}: {pred.est:.2f}")
