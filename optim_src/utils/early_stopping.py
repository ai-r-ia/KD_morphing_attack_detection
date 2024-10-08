class EarlyStopping:
    def __init__(self, logger, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.logger = logger

    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            self.logger.info(f"loss decreased ({self.best_loss:.6f} --> {loss:.6f}).")
        else:
            self.counter += 1
            self.logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
