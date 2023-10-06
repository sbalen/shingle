import numpy as np
from sklearn.decomposition import PCA

class ContextEncoder:
    def __init__(self, file):
        print("Loading Glove Model")
        glove_model = {}
        with open(file, 'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                glove_model[word] = embedding
        print(f"{len(glove_model)} words loaded!")
        self.broad_model = self.down_sample_embeddings(glove_model, 5)
        self.narrow_model = self.down_sample_embeddings(glove_model, 12)

    @staticmethod
    def down_sample_embeddings(model, dimensions):
        X = [value for value in model.values()]
        Y = [key for key in model.keys()]
        pca = PCA(n_components=dimensions)
        result = pca.fit_transform(X)
        
        # Normalizing column-wise
        result -= np.mean(result, axis=0)
        result /= np.std(result, axis=0)

        # Thresholding - setting values greater than zero to 1, others to 0
        result = np.where(result > 0, 1, 0)
        
        return dict(zip(Y, result))

    @staticmethod
    def to_context_hash(list_):
        return "".join([str(x) for x in list_])

    def get_contextgroup_code(self, word, broad=0):
        model = self.broad_model if broad else self.narrow_model
        return self.to_context_hash(model.get(word, []))
    
    def get_broad_context_code(self, word):
        return self.get_contextgroup_code(word, 1)
    
    def get_narrow_context_code(self, word):
        return self.get_contextgroup_code(word, 0)
