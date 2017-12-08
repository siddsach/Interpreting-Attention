from model import LangModel
from trainer import TrainLangModel
import math
from torchtext.data import Batch

trainer = TrainLangModel(batch_size=5, seq_len=3)

trainer.load_data()
text = trainer.raw_sentences[0].text

'''
print('NUMWORDS: {}'.format(len(text)))
TEXT = trainer.raw_sentences.fields['text']
TEXT.eos_token = None
text = text + ([TEXT.pad_token] * int(math.ceil(len(text) / trainer.batch_size) *
                                     trainer.batch_size - len(text)))
data = TEXT.numericalize(
    [text], device=-1, train=trainer.train)
print('After numericalizing')
print(type(data.data))
print(data.data.shape)
data = data.view(trainer.batch_size, -1).t().contiguous()
print('After converting dimensions based on batch size')
print(type(data.data))
print(data.data.shape)
dataset = None
#dataset = Dataset(examples=trainer.raw_sentences.examples, fields=[
#    ('text', TEXT), ('target', TEXT)])
length = math.ceil(len(trainer.raw_sentences[0].text) /
                         (trainer.batch_size * trainer.bptt_len))


while True:
    for i in range(0, length * trainer.bptt_len, trainer.bptt_len):
        seq_len = min(trainer.bptt_len, len(data) - 1 - i)
        b = Batch.fromvars(
            dataset, trainer.batch_size, train=trainer.train,
            text=data[i:i + seq_len],
            target=data[i + 1:i + 1 + seq_len])
        print(b.text.data, b.target.data)
    if not trainer.repeat:
        raise StopIteration
'''
'''
model = LangModel(trainer.sentence_field.vocab.__len__())
hidden = model.init_hidden(trainer.batch_size)
text = trainer.raw_sentences[0].text
TEXT = trainer.raw_sentences.fields['text']
TEXT.eos_token = None
text = text + ([TEXT.pad_token] * int(math.ceil(len(text) / trainer.batch_size) *
                              trainer.batch_size - len(text)))

if isinstance(text, tuple):
text, lengths = text
if TEXT.use_vocab:
if TEXT.sequential:
text = [[TEXT.vocab.stoi[x] for x in ex] for ex in text]
else:
text = [TEXT.vocab.stoi[x] for x in text]

if TEXT.postprocessing is not None:
text = TEXT.postprocessing(text, TEXT.vocab, True)
elif TEXT.postprocessing is not None:
    text = TEXT.postprocessing(text, True)
print('INPUT TO NUMERICALIZE')
print(type(text))
print(text)
text = TEXT.tensor_type(text)
data = TEXT.numericalize([text], device=-1, train=False)

print('DONE WITH MANUAL TEST')
for batch in trainer.batch_iterator:
    hidden = trainer.repackage_hidden(hidden)
    data = batch
    print("SHAPES")
    print(data.text.data)
    print(data.target.data)
    data, targets = batch.text, batch.target.view(-1)
    data = data.float()
    print("DATA")
    print(type(data.data))
    print(data.data.shape)
    print("HIDDEN")
    print([type(hidden[i].data) for i in range(2)])
    print([vec.data.shape for vec in hidden])
    output, hidden = model(data, hidden)
'''
trainer.get_iterator()
trainer.train()
