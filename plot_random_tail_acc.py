import numpy as np
import matplotlib.pyplot as plt

acc = [51.75,51.73,51.08,49.52,47.49,42.47,31.15]
acc = [51.75,51.73,51.55,50.71,49.11,47.32,00.00]
plt.figure(figsize=(5.5,3.8))
plt.plot(acc[:-1], marker='o')
plt.xticks([0,1,2,3,4,5],['No rand', '[k(x),8)', '[k(x)-1,8)', '[k(x)-2,8)', '[k(x)-3,8)', '[k(x)-4,8)'])
plt.xlabel('Tail expert randomization range')
plt.ylabel('Accuracy (%)')
plt.title('MMLU Top-8 Accuracy With Randomized Tail Experts')
plt.savefig('rand2.png', dpi=300)