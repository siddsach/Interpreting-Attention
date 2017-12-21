from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import subprocess
import pickle

data_dir = "data/gigaword"

max_files = 4
total_files = 22
path = "https://s3.amazonaws.com/gigaword/thread{}.txt"
filenames= [path.format(i+1) for i in range(total_files)]

num_docs = 0
locs = {}
all_docs = []

for i, a in enumerate(filenames):
    bashCommand = "wget " + a + " -P " + data_dir
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    reader = open(data_dir + "/thread{}.txt".format(i), 'r')
    docs = [el.split("\t")[1] for el in reader.read().split("\n")]
    locs[i] = (num_docs, num_docs + len(docs))
    all_docs += docs

    bashCommand = "rm" + data_dir + "/thread{}.txt".format(i)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    del docs

tf = CountVectorizer(stop_words = 'english', min_df = 20, max_df = 0.95, max_features = 500000)
X = tf.fit_transform(all_docs)

del all_docs

lda = LatentDirichletAllocation(n_components = 80, max_iter = 5,
        learning_method = 'online', random_state = 0)
assignments = lda.fit_transform(X)

def get_top_words(model, feature_names, n_top_words):
    results = ""
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
            for i in topic.argsort()[:-n_top_words - 1:-1]])

        results += message

topics = get_top_words(lda, tf.get_feature_names, 30)

topic_data = {"locs":locs, "topic_assignments": assignments}

pickle.dump(topic_data, open("gigaword_topic_model.pickle", "wb"))
f = open("topicwords.txt", 'w')
f.write(topics)
