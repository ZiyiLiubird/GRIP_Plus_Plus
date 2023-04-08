



class LinearScheduler():
    def __init__(self, start_lr, min_lr, max_steps, use_epochs=True):
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.max_steps = max_steps
        self.use_epochs = use_epochs
        
    def update(self, current_lr, epoch, frames=None):
        if self.use_epochs:
            steps = epoch
        else:
            steps = frames
        
        mul = max(0, self.max_steps - steps) / self.max_steps
        lr = self.min_lr + (self.start_lr - self.min_lr) * mul
        return lr