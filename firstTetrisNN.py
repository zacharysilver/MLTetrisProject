import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import joblib

def read_scores(file_path):

    with open(file_path, 'r') as file:
        scores = file.readlines()
        all_scores = []
        for line in scores:
            game_scores = eval(line.split(':')[1].strip())
            all_scores.extend(game_scores)
        return all_scores

def avg_starting_score(scores):
    starts = [(x, y) for x, y in scores if x == [0]*200]
    scores =[y for x, y in starts]
    mean_score = np.mean(scores)
    std_dev = np.std(scores)
    confidence_interval = 1.645 * std_dev / np.sqrt(len(scores))
    lower_bound = mean_score - confidence_interval
    upper_bound = mean_score + confidence_interval
    print(f"90% confidence interval: ({lower_bound}, {upper_bound})")
if __name__ == "__main__":
    #avg_starting_score(read_scores('tetris_scores.txt'))
    scores = read_scores('tetris_scores.txt')
    # Assuming scores are tuples of (input_features, target)
    X = np.array([score[0] for score in scores])
    y = np.array([score[1] for score in scores])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nn = joblib.load('tetris_nn_model.pkl')
    predictions = nn.predict(X_test)
    print(nn.score(X_test, y_test))
    # Plot the predicted vs actual scores
    plt.scatter(y_test, predictions)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)

    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.title('Actual vs Predicted Scores')
    # Plot the trendline
    z = np.polyfit(y_test, predictions, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), "r--")
    plt.show()
    
'''    file_path = 'tetris_scores.txt'
    scores = read_scores(file_path)
    # Assuming scores are tuples of (input_features, target)
    X = np.array([score[0] for score in scores])
    y = np.array([score[1] for score in scores])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the neural network
    nn = MLPRegressor(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)
    nn.fit(X_train, y_train)

    # Track the loss during training
    losses = nn.loss_curve_

    # Plot the loss over iterations
    plt.plot(losses)
    plt.title('Training Loss Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
    # Evaluate the model
    score = nn.score(X_test, y_test)
    print(f"Model accuracy: {score * 100:.2f}%")
    # Output the model parameters to a text file
    joblib.dump(nn, 'tetris_nn_model.pkl')'''
    