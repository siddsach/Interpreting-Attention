import pickle
import csv

data = pickle.load(open('/Users/siddharth/Downloads/mpqa_subj_labels.pickle', 'rb'))
subjective_file = open('subjective_sentences.csv', 'w')

writer = csv.writer(subjective_file)

writer.writerow(['text'])

for sent in data[0]:
    print(sent)
    writer.writerow([sent])

subjective_file.close()

