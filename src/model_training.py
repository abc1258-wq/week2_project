from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    models = {"Random Forest": model}

    return models, X_test, y_test