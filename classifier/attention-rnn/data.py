import pickle
import csv

out_path = 'sentence_subjectivity.csv'
filepath = '/Users/siddharth/flipsideML/ML-research/keysent_extraction/subjectivity/mpqa_subj_labels.pickle'

label_file = open(filepath, 'rb')
data = pickle.load(label_file)
train = dict()
train['sentences'] = data[0]
train['labels'] = data[1]

with open(out_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['sentences', 'labels'])
    for i in range(1, len(train['sentences'])):
        print('writing sentence {}'.format(i))
        sentence, label = train['sentences'][i], train['labels'][i]
        if type(sentence) != str or type(label) != bool or len(sentence) == 0:
            pass
        else:
            writer.writerow(data)
