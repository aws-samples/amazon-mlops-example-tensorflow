import unittest
import pandas as pd
import os
src_bucket = os.getenv("BUCKET_NAME")

class PreTraining(unittest.TestCase):
    
    def __init__(self,*args, **kwargs):
        super(PreTraining, self).__init__(*args, **kwargs)
        self.traindata = pd.read_pickle(f"s3://{src_bucket}/toxic_comments/train_data.pkl")
        self.trainlabels = pd.read_pickle(f"s3://{src_bucket}/toxic_comments/train_labels.pkl")
        
    def test_read_data(self):
        '''
        Ensure number of labels are equal to the training samples
        '''
        self.assertEqual(len(self.traindata),len(self.trainlabels))
    
    def test_multilabel(self):
        '''
        Ensure that there are 6 classes 
        '''
        self.assertEqual(self.trainlabels.shape[1],6)
    
    def test_multilabel_distribution(self):
        '''
        Ensure base rate is atleast 20% 
        '''
        labels = pd.DataFrame(self.trainlabels)
        positives = 0
        for i in range(6):
            positives = positives + sum(labels[i])
        baserate = round(positives/len(labels),3)
        self.assertGreater(baserate,.20)
    
    def test_seqlength_distribution(self):
        '''
        Ensure that comments are not disproportionately longer
        '''
        distinct_words = [len(row.split()) for row in self.traindata]
        bool_len = [i<100 for i in distinct_words]
        percentage = sum(bool_len)/len(bool_len)
        self.assertGreater(percentage,.80)
        
    def test_vocabsize(self):
        '''
        Ensure vocabulary size is not unexpected. This might blow up compute & training time
        '''
        vocab = set()
        for j in range(len(self.traindata)):
            for i in self.traindata[j].split():
                vocab.add(i) 
        self.assertLess(len(vocab),300000)
        
if __name__ == '__main__':
    unittest.main()
