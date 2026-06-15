import numpy
from numpy.typing import NDArray 
  
  

class Solution:
    def binary_cross_entropy(self, y_true: NDArray[np.float], y_pred: NDArray[np.float]) -> float:
        return round(-np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float], y_pred: NDArray[np.float]) -> float:
        print(y_true*np.log(y_pred))
        print(np.sum(y_true*np.log(y_pred)))
        return round(-np.sum(y_true*np.log(y_pred))/len(y_true), 4)