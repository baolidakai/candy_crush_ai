from matplotlib import pyplot as plt
import numpy as np

average_losses = []
with open('average_losses.txt', 'r') as fin:
  for line in fin:
    average_losses.append(float(line))
print(average_losses)

plt.plot(average_losses)
plt.savefig('avg_loss')

