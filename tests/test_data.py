# data class for test data

import numpy as np

class TestData:

    def __init__(self):
        pass

    @staticmethod
    def load_model_name():
        return 'faux_model'
    
    @staticmethod
    def load_class_names():
        return ['class_a', 'class_b', 'class_c']
    
    @staticmethod
    def load_data_shape():
        return (3, 3)
    
    @staticmethod
    def load_prior():
        return 1
    
    @staticmethod
    def load_data():
        return np.array([[100,10,5],
                         [10,100,5],
                         [5,10,100]])
    

    