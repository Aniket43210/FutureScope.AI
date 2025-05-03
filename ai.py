# Career Path Prediction ML Model
# This code implements a machine learning pipeline to predict suitable career paths
# based on student data (academics, hobbies, interests, and achievements)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class CareerPathPredictor:
    def __init__(self):
        """Initialize the career path prediction model."""
        self.model = None
        self.feature_names = None
        self.target_names = None

    def train(self, X, y, cv=5):
        """
        Train the career path prediction model with an integrated preprocessing pipeline.

        Parameters:
        -----------
        X : pandas DataFrame
            The input features containing student data
        y : pandas Series
            The target variable (career paths)
        cv : int
            Number of cross-validation folds

        Returns:
        --------
        self : CareerPathPredictor
            The trained model instance
        """
        # Define numerical and categorical features
        numerical_features = ['gpa', 'math_score', 'science_score', 'language_score', 
                              'social_studies_score', 'art_score', 'sports_score']
        categorical_features = ['favorite_subject', 'hobby_type', 'personality_type',
                                'learning_style', 'work_preference', 'leadership_role']
        self.feature_names = numerical_features + categorical_features
        self.target_names = sorted(y.unique())

        # Build a ColumnTransformer for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        # Create a pipeline that includes preprocessing and the classifier
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # Define hyperparameter grid for the classifier within the pipeline
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }

        # Split data into training and testing sets using raw features
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Perform grid search with cross-validation on the entire pipeline
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            scoring='f1_weighted'
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search

        print(f"Best parameters: {self.model.best_params_}")

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.target_names))

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.target_names,
                    yticklabels=self.target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

        # Feature importance analysis (if applicable)
        if hasattr(self.model.best_estimator_['classifier'], 'feature_importances_'):
            feature_importances = self.model.best_estimator_['classifier'].feature_importances_

            # The OneHotEncoder expands the categorical features.
            # Here we use placeholders for the expanded categorical feature names.
            n_num = len(numerical_features)
            n_total = len(feature_importances)
            n_cat = n_total - n_num
            placeholder_cat_features = [f"cat_feature_{i+1}" for i in range(n_cat)]
            all_feature_names = numerical_features + placeholder_cat_features

            # If dimension mismatch occurs, fallback to generic names
            if len(all_feature_names) != len(feature_importances):
                all_feature_names = [f"Feature_{i}" for i in range(len(feature_importances))]

            importance_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': feature_importances
            }).sort_values('Importance', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()

        return self

    def predict(self, X):
        """
        Predict career paths based on student data.

        Parameters:
        -----------
        X : pandas DataFrame
            The input features containing student data

        Returns:
        --------
        predictions : numpy array
            The predicted career paths
        probabilities : numpy array
            The probability for each career path
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Directly pass raw data to the pipeline for preprocessing and prediction
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        return predictions, probabilities

    def save_model(self, filepath):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)

        instance = cls()
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.target_names = model_data['target_names']

        return instance

# Example of how to generate synthetic data for testing/development
def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic student data for model development.
    This would be replaced with real data in production.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate

    Returns:
    --------
    X : pandas DataFrame
        Synthetic feature data
    y : pandas Series
        Synthetic target data
    """
    np.random.seed(42)

    # Generate synthetic features
    data = {
        'gpa': np.random.uniform(2.0, 10.0, n_samples),  # Updated max GPA to 10
        'math_score': np.random.uniform(50, 100, n_samples),
        'science_score': np.random.uniform(50, 100, n_samples),
        'language_score': np.random.uniform(50, 100, n_samples),
        'social_studies_score': np.random.uniform(50, 100, n_samples),
        'art_score': np.random.uniform(50, 100, n_samples),
        'sports_score': np.random.uniform(50, 100, n_samples),
        'favorite_subject': np.random.choice(['Math', 'Science', 'Arts', 'Language', 'Social Studies'], n_samples),
        'hobby_type': np.random.choice(['Technical', 'Creative', 'Physical', 'Social', 'Academic'], n_samples),
        'personality_type': np.random.choice(['Analytical', 'Creative', 'Social', 'Practical'], n_samples),
        'learning_style': np.random.choice(['Visual', 'Auditory', 'Kinesthetic'], n_samples),
        'work_preference': np.random.choice(['Individual', 'Team', 'Mixed'], n_samples),
        'leadership_role': np.random.choice(['Yes', 'No'], n_samples),
    }

    X = pd.DataFrame(data)

    # Generate synthetic target (career paths based on features)
    career_options = [
        'Engineering', 'Medicine', 'Business', 
        'Arts', 'Education', 'Computer Science',
        'Law', 'Social Work', 'Research'
    ]
    y = []
    for i in range(n_samples):
        if X.loc[i, 'math_score'] > 85 and X.loc[i, 'science_score'] > 80:
            if X.loc[i, 'personality_type'] == 'Analytical':
                career = np.random.choice(['Engineering', 'Computer Science', 'Research'], p=[0.5, 0.3, 0.2])
            else:
                career = np.random.choice(['Medicine', 'Engineering', 'Research'], p=[0.5, 0.3, 0.2])
        elif X.loc[i, 'language_score'] > 85 and X.loc[i, 'social_studies_score'] > 80:
            career = np.random.choice(['Law', 'Education', 'Business'], p=[0.4, 0.3, 0.3])
        elif X.loc[i, 'art_score'] > 85:
            career = np.random.choice(['Arts', 'Education', 'Business'], p=[0.6, 0.2, 0.2])
        elif X.loc[i, 'math_score'] > 80 and X.loc[i, 'language_score'] > 75:
            career = np.random.choice(['Business', 'Law', 'Computer Science'], p=[0.5, 0.3, 0.2])
        elif X.loc[i, 'science_score'] > 75 and X.loc[i, 'social_studies_score'] > 80:
            career = np.random.choice(['Medicine', 'Social Work', 'Education'], p=[0.4, 0.4, 0.2])
        else:
            career = np.random.choice(career_options)
        y.append(career)

    return X, pd.Series(y, name='career_path')

# Example usage
def main():
    # Generate synthetic data
    print("Generating synthetic student data...")
    X, y = generate_synthetic_data(n_samples=2000)

    print(f"Data shape: {X.shape}, Target shape: {y.shape}")
    print(f"Career distribution:\n{y.value_counts()}")

    # Create and train the model
    print("\nTraining the model...")
    predictor = CareerPathPredictor()
    predictor.train(X, y, cv=3)  # Use a smaller number of cross-validation folds for demonstration

    # Save the model
    predictor.save_model('career_predictor_model.joblib')

    # Example prediction
    print("\nMaking predictions for sample students...")
    sample_students = X.sample(5)
    predictions, probabilities = predictor.predict(sample_students)

    for i, (_, student) in enumerate(sample_students.iterrows()):
        print(f"\nStudent {i+1}:")
        print(f"  GPA: {student['gpa']:.2f}")
        print(f"  Favorite Subject: {student['favorite_subject']}")
        print(f"  Personality: {student['personality_type']}")
        print(f"  Predicted Career: {predictions[i]}")

        # Top 3 career recommendations with probabilities
        student_probs = probabilities[i]
        top3_indices = np.argsort(student_probs)[::-1][:3]

        print("  Top 3 Career Recommendations:")
        for idx in top3_indices:
            print(f"    {predictor.target_names[idx]}: {student_probs[idx]:.2f}")

if __name__ == "__main__":
    main()