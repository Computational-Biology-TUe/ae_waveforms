import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_rmse_distribution(avg_rmse, rmse_values, results_dir=None):
    """
    Plots a horizontal violin plot of RMSE values with the median and IQR.

    Parameters:
        avg_rmse (float): Average RMSE value.
        rmse_values (np.array): Array of RMSE values.
        results_dir (str): Directory to save the plot in.
    """
    # Create a figure and axis
    plt.figure(figsize=(8, 3))

    # Plot the violin plot
    sns.violinplot(data=rmse_values, orient='h', color='skyblue', inner="quart", linewidth=1.5)
    # Add a vertical line for the average RMSE
    plt.axvline(avg_rmse, color='red', linestyle='--', label=f'Avg RMSE = {avg_rmse:.4f}')

    # Customize the plot
    plt.xlabel('RMSE')
    plt.ylabel('Sample Density')
    plt.title('Distribution of RMSE Values')
    plt.legend()

    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the figure
    if results_dir:
        plt.savefig(os.path.join(results_dir, f"RMSE_distribution.png"), dpi=400, transparent=True)
    # Display the plot
    plt.show()

