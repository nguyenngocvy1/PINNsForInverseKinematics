import matplotlib.pyplot as plt
import numpy as np

labels = ['Average', 'Max', 'Min']
line_distances = [5.49, 9.93, 1.85]
circle_distances = [8.49, 20.45, 3.24]
line3D_distances = [7.33, 12.84, 1.28]
helix_distances = [8.41, 18.98, 1.78]

x = np.arange(len(labels))
width = 0.2  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(x - 1.5*width, line_distances, width, label='Line')
rects2 = ax.bar(x - 0.5*width, circle_distances, width, label='Circle')
rects3 = ax.bar(x + 0.5*width, line3D_distances, width, label='Line3D')
rects4 = ax.bar(x + 1.5*width, helix_distances, width, label='Helix')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Distances (mm)')
ax.set_title('Comparison for Loss Distances')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()
plt.savefig("model_12_29_14_42_[0.13155946135520935, 0.18043625354766846, 4.208344398648478e-05]\\distance_loss1.png")
plt.show()
