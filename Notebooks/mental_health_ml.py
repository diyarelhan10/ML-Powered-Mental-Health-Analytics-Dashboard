import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. LOAD & CLEAN DATA
df = pd.read_csv('../Data/mental_health_social_media_dataset.csv')

# Preprocessing: Drop identifiers and standardize formats
df_ml = df.drop(columns=['person_name'])
df_ml['date'] = pd.to_datetime(df_ml['date'])

# Encode Categorical features
le = LabelEncoder()
df_ml['gender'] = le.fit_transform(df_ml['gender'])
df_ml['platform'] = le.fit_transform(df_ml['platform'])
df_ml['target'] = df_ml['mental_state'].apply(lambda x: 0 if x == 'Healthy' else 1)

# 2. ML MODELING
features = ['age', 'daily_screen_time_min', 'social_media_time_min', 
            'sleep_hours', 'physical_activity_min', 
            'negative_interactions_count', 'positive_interactions_count']

X = df_ml[features]
y = df_ml['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. GENERATE VISUALS
# Graph 1: Feature Importance
plt.figure(figsize=(10, 6))
pd.Series(model.feature_importances_, index=features).sort_values().plot(kind='barh')
plt.title('Feature Importance')
plt.savefig('../Visuals/feature_importance.png')

# Graph 2: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('../Visuals/confusion_matrix.png')

# Graph 3: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_ml[features + ['target']].corr(), annot=True, cmap='coolwarm')
plt.title('Behavioral Correlation')
plt.savefig('../Visuals/correlation_heatmap.png')

print(f"Model Training Complete. Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}")
