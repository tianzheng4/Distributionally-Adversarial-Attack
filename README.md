# Distributionally Adversarial Attack

Recently, there exist many defense models against adversarial attacks. By an extensive evaluation, we figure out that one of the 
most effective defense methods is PGD adversarial training (*Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu* https://arxiv.org/abs/1706.06083)

Another potential provable defense method is:
https://github.com/locuslab/convex_adversarial

We design a new first-order attack algorithm by generalizing PGD on the space of data distributions and learning an adversarial distribution that maximally increases the generalization risk of a model, namely Distributionally Adversarial Attack (DAA). Our DAA attack achieves outstanding attack success rates on those state-of-the-art defense models. 
Our paper link is https://arxiv.org/abs/1808.05537 (Authors: *Tianhang Zheng, Changyou Chen, Kui Ren*) 

There are 2 attack versions, i.e., DAA-BLOB and DAA-DGF. Our code is written based on MadryLab's code: https://github.com/MadryLab/mnist_challenge

### Running the code
- `python blob_rand.py`: DAA-BLOB attack
- `python dgf_rand.py`: DAA-DGF attack
- `python pgd_rand.py`: PGD attack

CIFAR10 and Imagenet Coming Soon.

We also evaluate our attack against provable defense, and the code is also coming soon.

If there is any question, pls contact tzheng4@buffalo.edu
