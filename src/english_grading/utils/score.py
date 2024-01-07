import numpy as np
from sklearn.metrics import mean_squared_error


#  Precision-focused scoring 
def calculate_accuracies(true_values, predicted_values):
    """
    Calculates scores meticulously, capturing performance across multiple dimensions.

    Args:
        true_values (numpy.ndarray): A matrix containing the actual values.
        predicted_values (numpy.ndarray): A matrix containing the predicted values.

    Returns:
        tuple: A tuple harboring the overall mean score and a detailed list of individual scores.
    """

    # ️ Initialize a repository for scores ️
    scores = []  # Create an empty list to house scores

    #  Inspect each dimension 
    dimensional_count = true_values.shape[1]  # Determine the number of dimensions
    for i in range(dimensional_count):  # Explore each dimension
        true_vector = true_values[:, i]  # Extract the true values for this dimension
        predicted_vector = predicted_values[:, i]  # Extract the predicted values for this dimension

        #  Calculate the root mean squared error (RMSE) 
        score = mean_squared_error(true_vector, predicted_vector, squared=False)  # Measure accuracy using RMSE
        scores.append(score)  # Add the calculated score to the repository

    #  Calculate the mean score across all dimensions 
    overall_mean_score = np.mean(scores)  # Gather the collective wisdom of individual scores

    return overall_mean_score, scores  # Return both the summary and detailed scores for transparency ✨