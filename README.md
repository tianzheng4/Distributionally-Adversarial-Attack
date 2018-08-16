# Distributionally Adversarial Attack

Recently, there exist many defense models against adversarial attacks. By an extensive evaluation, we figure out that one of the 
most effective defense method is PGD adversarial training:
*Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu* <br> https://arxiv.org/abs/1706.06083.

Another potential provable defense method is:
https://github.com/locuslab/convex_adversarial

We design a new attack by generalizing PGD on the space of data distributions, by learning an adversarial distribution that maximally increases the generalization risk of a model. Our attack is called Distributionally Adversarial Attack (DAA). Our paper is submitted on Arxiv, and it will show up tomorrow. (link)

There are 2 versions for our DAA attack, i.e., DAA-BLOB and DAA-DGF. Our code is written based on https://github.com/MadryLab/mnist_challenge

### Running the code
- `python blob_rand.py`: DAA-BLOB attack
- `python dgf_rand.py`: DAA-DGF attack

CIFAR10 and Imagenet Coming Soon.

We also evaluate our attack against provable defense, and the code will also come soon.
