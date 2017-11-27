import jsonpickle
import cPickle
import json
with open('data/nottingham.pickle', 'r') as f:
    x = cPickle.load(f)
    
xs = jsonpickle.encode(x)
with open('nottingham.json', 'w') as f:
    json.dump(xs, f)