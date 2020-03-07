from random import randint
import pylab as plt
from matplotlib_venn import venn2

A = [2, 4, 6]

B = [1, 2, 3, 4]

A_count = 0

B_count = 0

AB_count = 0

n = 1000

for i in range(0, n):

    toss = randint(1, 6)

    if toss in A:
        A_count += 1

    if toss in B:
        B_count += 1

    if toss in A and toss in B:
        AB_count += 1


P_A = A_count/n

P_B = B_count/n

P_AB = AB_count/n

print('Prob(A):')
print(P_A)

print('Prob(B):')
print(P_B)

print('Prob(AB):')
print(P_AB)

print('Prob(A) * Prob(B):')
print(P_A * P_B)

v = venn2(subsets=(round(P_A - P_AB, 3), round(P_B - P_AB, 3), round(P_AB, 3)))

plt.title("Venn diagram of a fair toss")

plt.show()

