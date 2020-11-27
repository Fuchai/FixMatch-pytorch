import torch.nn as nn
import torch

class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.l1=nn.Linear(1,10)
        self.l2=nn.Linear(10,10)
        self.l3=nn.Linear(10,1)

    def forward(self, input):
        return self.l3(self.l2(self.l1(input)))


if __name__ == '__main__':
    print("Hello")
    input = torch.Tensor([28])
    a=A()
    b=A()
    out_a = a(input)
    out_b=b(input)
    out = out_a+out_b
    print("out", out_b)

    target = torch.Tensor([10])
    loss = nn.functional.l1_loss(out, target)
    print("loss", loss)
    print("a grad before", a.l1.weight.grad)
    print("b grad before", b.l1.weight.grad)
    loss.backward()
    b.zero_grad()
    print("a grad", a.l1.weight.grad)
    print("b grad", b.l1.weight.grad)
    loss.backward()
    print("a grad", a.l1.weight.grad)
    print("b grad", b.l1.weight.grad)