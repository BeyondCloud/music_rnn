import cPickle
from model import Model, NottinghamModel
from rnn import DefaultConfig
if __name__ == '__main__':
	with open('test.config', 'r') as f: 
	        config = cPickle.load(f)
	        print config