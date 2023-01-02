import torch
from torch import nn

scores_matrix = torch.rand(2, 2)
softmax = nn.Softmax(dim=0)
preds = softmax(scores_matrix).T
#
# print(preds)
# print(preds.shape)
# #
# labels = torch.tensor([1, 0])
# print(labels)
# # print(labels.shape)
# # #
# loss = nn.CrossEntropyLoss()
# print(loss(labels, preds))

# # Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(2, 2, requires_grad=True)
print(input)
target = torch.empty(2, dtype=torch.long).random_(1)
print(target)
output = loss(preds, target)
print(output)
# output.backward()
# # Example of target with class probabilities
# input = torch.randn(3, 5, requires_grad=True)
# print(input)
# target = torch.randn(3, 5).softmax(dim=1)
# print(target)
# output = loss(input, target)
# output.backward()
