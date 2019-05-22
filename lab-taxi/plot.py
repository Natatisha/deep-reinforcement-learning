import matplotlib.pyplot as plt


def plot_rewards(avg_rewards):
    plt.figure(figsize=(20, 10))
    plt.xlabel('episodes')
    plt.ylabel('average reward')
    plt.plot(avg_rewards)
    plt.show()
