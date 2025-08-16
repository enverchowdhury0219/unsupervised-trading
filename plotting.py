import matplotlib.pyplot as plt

def plot_curve(curve, title: str):
    ax = curve.plot(title=title)
    ax.set_ylabel("Equity (start = 1.0)")
    plt.tight_layout()
    plt.show()
