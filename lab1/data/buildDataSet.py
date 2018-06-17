import gzip
import tqdm
import numpy as np
import pickle
import json


SMALL_TRAIN_SIZE = 800

class FullBeerDataset(object):

    def __init__(self, mode, max_length=500, stem='data/beer_review/reviews.aspect'):
        aspect = 'overall'
        self.name = mode
        self.dataset = []
        self.max_length = max_length
        self.aspects_to_num = {'appearance':0, 'aroma':1, 'palate':2,'taste':3, 'overall':4}
        self.class_map = {0: 0, 1:0, 2:0, 3:0, 4:0,
                        5:1, 6:1, 7:1 , 8:2, 9:2, 10:2}
        self.name_to_key = {'train':'train', 'dev':'heldout', 'test':'heldout'}
        self.class_balance = {}
        with gzip.open(stem+str(self.aspects_to_num[aspect])+'.'+self.name_to_key[self.name]+'.txt.gz') as gfile:
            lines = gfile.readlines()
            lines = zip( range(len(lines)), lines)

            if self.name == 'dev':
                lines = lines[:5000]
            elif self.name == 'test':
                lines = lines[5000:10000]
            elif self.name == 'train':
                lines = lines[0:20000]

            for indx, line in tqdm.tqdm(enumerate(lines)):
                uid, line_content = line
                sample = self.processLine(line_content, self.aspects_to_num[aspect])

                if not sample['y'] in self.class_balance:
                    self.class_balance[ sample['y'] ] = 0
                self.class_balance[ sample['y'] ] += 1
                sample['uid'] = uid
                self.dataset.append(sample)
            gfile.close()
        print ("Class balance", self.class_balance)

    ## Convert one line from beer dataset to {Text, Tensor, Labels}
    def processLine(self, line, aspect_num):
        labels = [ float(v) for v in line.split()[:5] ]
        label = int(self.class_map[ int(labels[aspect_num] *10) ])
        text_list = line.split('\t')[-1].split()[:self.max_length]
        text = " ".join(text_list)
        sample = {'text':text, 'y':label}
        return sample

if __name__ == '__main__':
    train_data = FullBeerDataset('train')
    pickle.dump(train_data.dataset, open('data/train.p','wb'))
    dev_data = FullBeerDataset('dev')
    pickle.dump(dev_data.dataset, open('data/dev.p','wb'))
    test_data = FullBeerDataset('test')
    pickle.dump(test_data.dataset, open('data/test.p','wb'))