import matplotlib.pyplot as plt


def plot_metrics(results1, results2):
    trunks = list(results1.keys())

    fig, axs = plt.subplots(3, figsize=(6, 12))

    for i, metric_name in enumerate(["Chamfer Distance", "Coverage", "Minimum Matching Distance"]):
        metric_values1 = []
        metric_values2 = []
        for trunk in trunks:
            trunk_metrics1 = results1[trunk]
            trunk_metrics2 = results2[trunk]
            trunk_metric_values1 = [metrics[i] for metrics in trunk_metrics1]
            trunk_metric_values2 = [metrics[i] for metrics in trunk_metrics2]
            metric_values1.append(trunk_metric_values1)
            metric_values2.append(trunk_metric_values2)

        axs[i].plot(trunks, metric_values1, marker='o', label='pi-GAN')
        axs[i].plot(trunks, metric_values2, marker='o', label='EG3D')
        axs[i].set_title(metric_name)
        axs[i].set_xlabel("Trunk")
        axs[i].set_ylabel(metric_name)
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()