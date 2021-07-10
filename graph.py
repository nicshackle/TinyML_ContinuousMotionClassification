import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filename = "walking.csv"

df = pd.read_csv(filename)

sample_size = 119

index = range(1, len(df['aX']) + 1)
small_index = range(sample_size)

plt.rcParams["figure.figsize"] = (20,10)

plt.plot(index, df['aX'], 'g.', label='Ax', linestyle='solid', marker=',')
plt.plot(index, df['aY'], 'b.', label='Ay', linestyle='solid', marker=',')
plt.plot(index, df['aZ'], 'r.', label='Az', linestyle='solid', marker=',')
plt.title("Acceleration data (all samples)")
plt.xlabel("Sample #")
plt.ylabel("Acceleration (G)")
plt.legend()
plt.show()

plt.plot(small_index, df['aX'][:sample_size], 'g.', label='x', linestyle='solid', marker=',')
plt.plot(small_index, df['aY'][:sample_size], 'b.', label='y', linestyle='solid', marker=',')
plt.plot(small_index, df['aZ'][:sample_size], 'r.', label='z', linestyle='solid', marker=',')
plt.title("Acceleration (one sample)")
plt.xlabel("Sample #")
plt.ylabel("Acceleration (G)")
plt.legend()
plt.show()

plt.plot(index, df['gX'], 'g.', label='Gx', linestyle='solid', marker=',')
plt.plot(index, df['gY'], 'b.', label='Gy', linestyle='solid', marker=',')
plt.plot(index, df['gZ'], 'r.', label='Gz', linestyle='solid', marker=',')
plt.title("Gyro data (all samples)")
plt.xlabel("Sample #")
plt.ylabel("Rotation (dps)")
plt.legend()
plt.show()

plt.plot(small_index, df['gX'][:sample_size], 'g.', label='Gx', linestyle='solid', marker=',')
plt.plot(small_index, df['gY'][:sample_size], 'b.', label='Gy', linestyle='solid', marker=',')
plt.plot(small_index, df['gZ'][:sample_size], 'r.', label='Gz', linestyle='solid', marker=',')
plt.title("Gyro data (one sample)")
plt.xlabel("Sample #")
plt.ylabel("Rotation (dps)")
plt.legend()
plt.show()

