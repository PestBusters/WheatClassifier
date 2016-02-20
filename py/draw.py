import matplotlib.pyplot as plt
from matplotlib import gridspec

# http://matplotlib.org/users/gridspec.html

fig = plt.figure(figsize=(8, 8))

# gridspec inside gridspec
outer_grid = gridspec.GridSpec(4, 4, wspace=0.0, hspace=0.0)

for i in range(0, 1):
    # i = i + 7
    inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)

    ax = plt.Subplot(fig, inner_grid[0])
    # ax.imshow(train_imgs[i])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)

    ax = plt.Subplot(fig, inner_grid[1])
    # ax.plot(train_f[i])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)

plt.show()

# ax2 = plt.subplot2grid((3,3), (1, 0), colspan=2)
# ax3 = plt.subplot2grid((3,3), (1, 2), rowspan=2)
# plt.show()
