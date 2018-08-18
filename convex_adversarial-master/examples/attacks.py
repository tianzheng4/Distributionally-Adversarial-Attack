import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from convex_adversarial import robust_loss

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def mean(l): 
    return sum(l)/len(l)

def _fgs(model, X, y, epsilon): 
    opt = optim.Adam([X], lr=1e-3)
    out = model(X)
    ce = nn.CrossEntropyLoss()(out, y)
    err = (out.data.max(1)[1] != y.data).float().sum()  / X.size(0)

    opt.zero_grad()
    ce.backward()
    eta = X.grad.data.sign()*epsilon
    
    X_fgs = Variable(X.data + eta)
    err_fgs = (model(X_fgs).data.max(1)[1] != y.data).float().sum()  / X.size(0)
    return err, err_fgs

def fgs(loader, model, epsilon, verbose=False, robust=False): 
    return attack(loader, model, epsilon, verbose=verbose, atk=_fgs,
                  robust=robust)


def _pgd(model, X, y, epsilon, niters=100, alpha=0.01): 
    out = model(X)
    ce = nn.CrossEntropyLoss()(out, y)
    err = (out.data.max(1)[1] != y.data).float().sum()  / X.size(0)

    X_pgd = Variable(X.data, requires_grad=True)
    for i in range(niters): 
        opt = optim.Adam([X_pgd], lr=1e-3)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = alpha*X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        
        # adjust to be within [-epsilon, epsilon]
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum() / X.size(0)
    return err, err_pgd
    
def _svgd(model, X, y, epsilon, niters=100, alpha=0.01): 
    out = model(X)
    ce = nn.CrossEntropyLoss()(out, y)
    err = (out.data.max(1)[1] != y.data).float().sum()  / X.size(0)

    X_pgd = Variable(X.data, requires_grad=True)
    for i in range(niters): 
        opt = optim.Adam([X_pgd], lr=1e-3)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        
        grad = X_pgd.grad.data
        Kxy, dxkxy = svgd_kernel(X_pgd.data)
        
        X_shape = list(X_pgd.data.size())
        svgd = -(torch.mm(Kxy, -grad.view(-1, X_shape[1]*X_shape[2]*X_shape[3])) + dxkxy)/float(X_shape[0])
        
        eta = alpha*(0.05*svgd.view(X_shape[0], X_shape[1], X_shape[2], X_shape[3])+grad).sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        
        # adjust to be within [-epsilon, epsilon]
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum() / X.size(0)
    return err, err_pgd
    
def _sgld(model, X, y, epsilon, niters=100, alpha=0.01): 
    out = model(X)
    ce = nn.CrossEntropyLoss()(out, y)
    err = (out.data.max(1)[1] != y.data).float().sum()  / X.size(0)

    X_pgd = Variable(X.data, requires_grad=True)
    for i in range(niters): 
        opt = optim.Adam([X_pgd], lr=1e-3)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        
        grad = X_pgd.grad.data
        Kxy, dxkxy = wgf_kernel(X_pgd.data)
        
        X_shape = list(X_pgd.data.size())

        eta = alpha*(-0.0001*dxkxy.view(X_shape[0], X_shape[1], X_shape[2], X_shape[3])+grad).sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        
        # adjust to be within [-epsilon, epsilon]
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum() / X.size(0)
    return err, err_pgd

def svgd_kernel(Theta):
    Theta_shape = list(Theta.size()) 
    theta = Theta.view(-1, Theta_shape[1]*Theta_shape[2]*Theta_shape[3])

    pairwise_dists = pairwise_dist(theta, theta)
    theta_shape = list(theta.size())
    h = torch.median(pairwise_dists)
    h_square = 0.5*torch.div(h, torch.log(torch.tensor([float(theta_shape[0])]).cuda()))
    Kxy = torch.exp(-0.5*torch.div(pairwise_dists, h_square))
    
    dxkxy = -torch.mm(Kxy, theta)
    sumkxy = torch.sum(Kxy, dim=1, keepdim=True)
    dxkxy = dxkxy + theta*sumkxy.expand((theta_shape[0], theta_shape[1]))
    dxkxy = torch.div(dxkxy, h_square)
    
    return (Kxy, dxkxy)
    
def wgf_kernel(Theta):
    Theta_shape = list(Theta.size()) 
    theta = Theta.view(-1, Theta_shape[1]*Theta_shape[2]*Theta_shape[3])

    pairwise_dists = pairwise_dist(theta, theta)
    theta_shape = list(theta.size())
    h = torch.median(pairwise_dists)
    h_square = 0.5*torch.div(h, torch.log(torch.tensor([float(theta_shape[0])]).cuda()))
    Kxy = torch.exp(-0.5*torch.div(pairwise_dists, h_square))
    Kxy = (torch.div(pairwise_dists, 2*h_square) - 1.0) * Kxy
    
    dxkxy = -torch.mm(Kxy, theta)
    sumkxy = torch.sum(Kxy, dim=1, keepdim=True)
    dxkxy = dxkxy + theta*sumkxy.expand((theta_shape[0], theta_shape[1]))
    
    return (Kxy, dxkxy)
    
def pairwise_dist (A, B):
    D = A.pow(2).sum(1, keepdim = True) + B.pow(2).sum(1, keepdim = True).t() - 2 * torch.mm(A, B.t())
    return torch.clamp(D, 0.0)

def pgd(loader, model, epsilon, niters=100, alpha=0.01, verbose=False,
        robust=False):
    return attack(loader, model, epsilon, verbose=verbose, atk=_pgd,
                  robust=robust)
                  
def svgd(loader, model, epsilon, niters=100, alpha=0.01, verbose=False,
        robust=False):
    return attack(loader, model, epsilon, verbose=verbose, atk=_svgd,
                  robust=robust)
                  
def dgf(loader, model, epsilon, niters=100, alpha=0.01, verbose=False,
        robust=False):
    return attack(loader, model, epsilon, verbose=verbose, atk=_sgld,
                  robust=robust)
                  
def attack(loader, model, epsilon, verbose=False, atk=None,
           robust=False):
    
    total_err, total_fgs, total_robust = [],[],[]
    if verbose: 
        print("Requiring no gradients for parameters.")
    for p in model.parameters(): 
        p.requires_grad = False
    
    for i, (X,y) in enumerate(loader):
        #print('batch: ', i)
        X,y = Variable(X.cuda(), requires_grad=True), Variable(y.cuda().long())

        if y.dim() == 2: 
            y = y.squeeze(1)
        
        if robust: 
            robust_ce, robust_err = robust_loss_batch(model, epsilon, X, y, False, False)

        err, err_fgs = atk(model, X, y, epsilon)
        
        total_err.append(err)
        total_fgs.append(err_fgs)
        if robust: 
            total_robust.append(robust_err)
        if verbose: 
            if robust: 
                print('err: {} | attack: {} | robust: {}'.format(err, err_fgs, robust_err))
            else:
                print('err: {} | attack: {}'.format(err, err_fgs))
    
    if robust:         
        print('[TOTAL] err: {} | attack: {} | robust: {}'.format(mean(total_err), mean(total_fgs), mean(total_robust)))
    else:
        print('[TOTAL] err: {} | attack: {}'.format(mean(total_err), mean(total_fgs)))
    return total_err, total_fgs, total_robust

