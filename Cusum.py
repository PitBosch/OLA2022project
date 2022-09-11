class Cusum:
    def __init__(self, M:int, eps:float, h:float):
        """ Initialize the relevant variables 
            M : number of sample stored before change detection kicks 
            eps : tollerance for change detection"""
        self.M = M
        self.eps = eps
        self.h = h
        self.t = 1
        self.last_change_t = 1
        self.reference = 0.
        self.g_plus = 0.
        self.g_minus = 0.

    def update(self, sample: float) -> bool :
        """ Takes a 1d sample and return True if a detection was flagged """
        self.t += 1
        if self.t <= self.M:
            self.reference += sample/self.M
            return False
        else:
            self.reference = (self.reference*(self.t-1) + sample)/self.t
            s_plus = (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps
            self.g_plus = max(0, self.g_plus + s_plus)
            self.g_minus = max(0, self.g_minus + s_minus)
            return self.g_plus > self.h or self.g_minus > self.h
    def reset(self, change_t):
        """ Reset all the relevant variable"""  
        self.last_change_t = change_t
        self.g_minus = 0
        self.g_plus = 0
        self.t = 1
        self.reference = 0