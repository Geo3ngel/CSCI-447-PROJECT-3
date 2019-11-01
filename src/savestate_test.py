class temp1():
    def __init__(self):
        self.item = "Test1"
        
class temp2():
    def __init__(self):
        self.not_item = "Test2."
        
import save_state

tmp = temp1()

save_state.save_state(tmp, "test_obj")