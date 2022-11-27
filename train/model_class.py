"""
 model class define
 mainly include model class for flat ner
"""
from processors.processor import LEBertProcessor, BertProcessor
from models.ner_model import LEBertSoftmaxForNer, LEBertCrfForNer, \
    BertCrfForNer, BertSoftmaxForNer, BertLSTMSoftmaxForNer, BertLSTMCrfForNer


MODEL_CLASS = {
    'lebert-softmax': LEBertSoftmaxForNer,
    'lebert-crf': LEBertCrfForNer,
    'bert-softmax': BertSoftmaxForNer,
    'bert-crf': BertCrfForNer,
    'bert-lstm-softmax': BertLSTMSoftmaxForNer,
    'bert-lstm-crf': BertLSTMCrfForNer
}
PROCESSOR_CLASS = {
    'lebert-softmax': LEBertProcessor,
    'lebert-crf': LEBertProcessor,
    'bert-softmax': BertProcessor,
    'bert-crf': BertProcessor,
    'bert-lstm-softmax': BertProcessor,
    'bert-lstm-crf': BertProcessor
}