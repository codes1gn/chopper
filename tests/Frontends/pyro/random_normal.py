import pyro
import torch

from chopper.pytorch import *

def random_normal_test(shape):
    class Test(torch.nn.Module):
        
        @backend("IREE")
        @set_target_ir("mhlo")
        @annotate_arguments([
            None,
            (shape, torch.float32),
            (shape, torch.float32),
            (shape, torch.float32),
            (shape, torch.float32)
        ])
        def forward(self, a, b, c, d):
            x = pyro.sample("my_sample", pyro.distributions.Normal(a, b))#, sample_shape = [2])
            y = pyro.sample("my_sample2", pyro.distributions.Normal(c, d))#, sample_shape = [2])
            
            z = x + y
            return z
        
        def ref_forward(self, loc, scale):
            x = pyro.sample("my_sample", pyro.distributions.Normal(loc, scale)) #)
            return x
        
        
    test = Test()
    loc = torch.zeros(shape).detach()
    scale = torch.ones(shape).detach()
    
    a = c = torch.zeros(shape).detach()
    b = d = torch.ones(shape).detach()
    print("loc = ", loc)
    print("scale = ", scale)
    act_res = test(a, b, c, d)
    ref_res = test.ref_forward(loc, scale)
    
    print("act_res = ", act_res)
    print("ref_res = ", ref_res)
    
random_normal_test((2,3))
    

def sample_grad():
    loc = torch.tensor([0.0,1.0, 2.0, 3.0]).requires_grad_(True)
    scale = torch.tensor([1.0, 1.0, 1.0, 1.0]).requires_grad_(True)
    z = torch.zeros(4)
    o = torch.ones(4)
    a = pyro.sample("my_sample", pyro.distributions.Normal(z, o))
    # a = torch.distributions.normal.Normal(z, o).sample()
    print(a)
    
    a = scale * a + loc
    print(a)

    a.sum().backward()
    print(f"after backward, loc.grad = {loc.grad} and scale.grad = {scale.grad}")
    # loc.grad.zero_()
    # scale.grad.zero_()


# sample_grad()

def sample_batch_test():
    # loc = 0.0
    # scale = 1.0
    # x = pyro.distributions.Normal(loc, scale)
    # print(x)
    # print(type(x))
    
    seed = 0
    torch.manual_seed(seed)   

    a = pyro.sample("my_sample1", pyro.distributions.Normal(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([1.0])))
    b = pyro.sample("my_sample2", pyro.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])))
    c = pyro.sample("my_sample3", pyro.distributions.Normal(torch.tensor([1.0]), torch.tensor([1.0])))
    d = pyro.sample("my_sample4", pyro.distributions.Normal(torch.tensor([0.0]), torch.tensor([2.0])))
    e = pyro.sample("my_sample5", pyro.distributions.Normal(torch.tensor([[0.0, 1.0, 2.0],[1.0, 2.0, 3.0]]), torch.tensor([1.0])))
    f = pyro.sample("my_sample6", pyro.distributions.Normal(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([[1.0, 2.0, 3.0],[1.0, 2.0, 3.0]])))

    print("a = ", a)
    print("b = ", b)
    print("c = ", c)
    print("d = ", d)
    print("e = ", e)
    print("f = ", f)
    print("====================================")
    
