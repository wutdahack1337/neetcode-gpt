import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        
        mx = np.max(z)
        
        sm = 0
        for x in z:
            sm += np.exp(x - mx);

        print(mx, sm);

        z = [np.exp(x - mx)/sm for x in z]

        return np.round(z, 4)
