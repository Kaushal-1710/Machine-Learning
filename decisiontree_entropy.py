import os
# Dynamically add Graphviz to PATH so that Python can execute Graphviz 'dot' command
graphviz_path = r'C:\Program Files\Graphviz\bin'
if graphviz_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + graphviz_path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from dtreeviz.trees import dtreeviz
from sklearn.tree import DecisionTreeClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy')  
# ID3 uses information gain, which is based on entropy
model.fit(X_train, y_train)

# Visualize the decision tree
viz = dtreeviz(model, X, y, feature_names=iris.feature_names, class_names=list(iris.target_names))

# Save the visualization to an SVG file
viz.save("decision_tree.svg")
print("Decision tree visualization successfully saved to 'decision_tree.svg'")

# Attempt to open the visualization in browser/default viewer
viz.view()

