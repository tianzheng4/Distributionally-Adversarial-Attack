# Distributionally Adversarial Attack
Update: I decided to extend the paper to a long version later to explain more the original paper and detail the **Adaptive part**. I expect to put the long version on Arxiv in one/two months. Pls stay tuned.

Update: I just found that the formulas in the original paper have few minor issues, which might cause a little confusion.
I will update the paper on Arxiv when I am available. But, it is worth noting that the python code is correct. 

Update: The convex folder is updated, you can evaluate DAA on provable network by the main_attack.py in the its examples folder.

Recently, many defense models against adversarial attacks existed. By an extensive evaluation, we figure out that one of the 
most effective defense methods is PGD adversarial training (*Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu* https://arxiv.org/abs/1706.06083)

Another potential provable defense method is:
https://github.com/locuslab/convex_adversarial

We design a new first-order attack algorithm by generalizing PGD on the space of data distributions and learning an adversarial distribution that maximally increases the generalization risk of a model, namely Distributionally Adversarial Attack (DAA). Our DAA attack achieves outstanding attack success rates on those state-of-the-art defense models. 
Our paper link is https://arxiv.org/abs/1808.05537 (Paper Authors: *Tianhang Zheng, [Changyou Chen](https://cse.buffalo.edu/~changyou/), [Kui Ren](https://www.acsu.buffalo.edu/~kuiren/)*) 

There are 2 attack versions, i.e., DAA-BLOB and DAA-DGF. Our code is written based on [MadryLab's code](https://github.com/MadryLab/mnist_challenge)


### Running the code
Since the files in models are not fully uploaded to Github, pls download madry's models using `python3 fetch_model.py secret` 
- `python blob_rand.py`: DAA-BLOB attack
- `python dgf_rand.py`: DAA-DGF attack
- `python pgd_rand.py`: PGD attack
- `python mi_pgd_rand.py`: PGD variant of Momentum Iterative attack

Python3.5/3.6 is suggusted.

Our MNIST result shows on MadryLab's white-box leaderboard :-) =>  https://github.com/MadryLab/mnist_challenge

Our CIFAR10 result shows on MadryLab's white-box leaderboard :-) =>  https://github.com/MadryLab/cifar10_challenge


We also evaluate our attack against provable defense, and the code is in the directory "convex_adversarial-master".

If there is any question, pls contact tzheng4@buffalo.edu
