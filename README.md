# Human_Activity_Recognition
Classifying sequences of accelerometer data recorded by smart phones into known well-defined movements

Input Data
obtained input data from UCI Machine Learning Repository
Number of traingin Examples: 10,299
Features: 561
Test: 50%
Validation: 22%
Test: 28%

Preprocessing
Standarization of input data

Principal Component Analysis
PC Scores Estimated(Scree Plot)
Found the most optimal at 200

-Reduced computational cost
-Avoid overfitting
-Give more important features
-Dimensionality Reduction
-Elimation of Redundant Features

Steps Mathematically:
-find the mean
-shift the origin to the mean
-Rotate the axis to find the direction of maximum variance which will be the loading score corresponding to the first PC
-Do the same for rest of the PC
-The direction of maximum variance is the Eigen Vector of the covariance matrix of (feature x feature)
-Its value is proportional to eigen value.
-variance = (eigen value)/sample space size
-Eigen value id actualy the sum of projection of all the training examples along the direction of PC or eigen vector
-Transform the original matrix (n*m) by multiplying to it transformation matrix (d*n) where d is the number of selected PCs, here 200.
-obtained input after dimensionality reduction (d*m) where m i the no. of training examples, n is the original number of features.

Application of Support Vector Machine
-Used one vs all with linear kernel for classification
-Used in case of constrained optimization problem
-Equality between prim and dual form by using Lagrange's equation which is established at the KKT condition
-Benifits of representation in the form of inner product:
     (i) We can prove how the decision boundry depends on very few training examples, which we call "support vectors"
     (ii) We would be able to apply kernel trick, which can assist in classification of the different classes without going into very
           high dimensional space and the reducing the computational cost tremendously.
Note-
The decision boundry is hyperplane(one dimension less than the actual dimension representated by the number of features).
Use of "Geometric Margin" as large margin classifier.
Cost J regulaeized by the square of norm of W, W is the weight matrix.

Error Metrics
Use of tradeoff between Precision and Recall i.e. F1 Score
Precision= TP/(TP+FP)
Recall= TP/(TP+FN)
F=(2*precision*Recall) / (Precision+Recall)

Accuracy:
Training:98
Test:95~96
Validation:95~96
