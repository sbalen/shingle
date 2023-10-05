import numpy as np
from sklearn.decomposition import PCA
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

class ContextEncoder:
  def __init__(self, File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    self.broad_model = self.down_sample_embeddings(model, 5)
    self.narrow_model = self.down_sample_embeddings(model, 12)

# Takes a embedding model, returns a strongly downcoded one
@staticmethod
def down_sample_embeddings(model, dimensions):
  X = [value for value in model.values()]
  Y = [key for key in model.keys()]
  pca = PCA(n_components=dimensions)
  result = pca.fit_transform(X)

  #This should be improved, the normalisation should be columnwise
  average_value = np.mean(result)
  result = np.where(result > average_value, 1, 0)
  return dict(zip(Y, result))  

  @staticmethod
  def to_context_hash(list):
    return "".join([str(x) for x in list])
  
  def get_contextgroup_code(word, broad=0):
    #if(word not in  model): #this is a fix for privily which i now replaced for softly for complexity purposes
    #  return to_context_hash(model[word])
    if(broad):
      return to_context_hash(broad_model[word])
    else:
      return to_context_hash(narrow_model[word])
  
  def get_broad_context_code(word):
    return get_contextgroup_code(word, 1)
  
  def get_broad_narrow_code(word):
    return get_contextgroup_code(word, 0)
