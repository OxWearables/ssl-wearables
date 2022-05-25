class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, age, gender, ethnicity):
        mse, crossEntropy = MSELossFlat(), CrossEntropyFlat()

        loss0 = mse(preds[0], age)
        loss1 = crossEntropy(preds[1], gender)
        loss2 = crossEntropy(preds[2], ethnicity)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0 * loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1 * loss1 + self.log_vars[1]

        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2 * loss2 + self.log_vars[2]

        return loss0 + loss1 + loss2

