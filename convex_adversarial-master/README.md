# Evaluation on Provably robust neural networks

To run the attacks, open the examples directory:
```
python main_attacks.py --mnist --epsilon=0.15
```

The results include attack success rates of PGD, DAA-BLOB, DAA-DGF (output is in this order)

attack success rate = 1 - classification accuracy under the attack