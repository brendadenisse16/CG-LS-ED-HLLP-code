from typing import Dict

class Parameters:
    DEFAULTS: Dict[str, float] = {
        'N': 10,
        'P': 6,
        'ALPHA': 0.2,
        'R': 1.7,
        'V': 0.01,
        'PERC': 1.0,
        'EPSILON': 0.00005
    }
    def __init__(self, 
                 n: int = DEFAULTS.get('N'), 
                 p: int = DEFAULTS.get('P'), 
                 alpha: float = DEFAULTS.get('ALPHA'), 
                 r: float = DEFAULTS.get('R'), 
                 v: float = DEFAULTS.get('V'), 
                 perc: float = DEFAULTS.get('PERC'),
                 epsilon: float = DEFAULTS.get('EPSILON')
                 ) -> None:
        """Initializes parameters with given inputs or defaults if no input is given.

        :param n: ...
        :param p: ...
        :param alpha: ...
        :param r: ...
        :param v: ...
        :param perc: ...
        """
        self.n = n
        self.p = p
        self.alpha = alpha
        self.r = r
        self.v = v
        self.perc = perc
        self.epsilon = epsilon