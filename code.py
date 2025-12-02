import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import requests
import time
from geopy.geocoders import Nominatim

df = pd.read_csv("C:\\Users\\ARTUR\\Desktop\\AI-model\\earthquake_1995-2023(ed).csv")
df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M')

df['year'] = df['date_time'].dt.year
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day
df['hour'] = df['date_time'].dt.hour
df['dayofweek'] = df['date_time'].dt.dayofweek

categorical_cols = ['magType', 'alert', 'tsunami']
df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], drop_first=True)

exclude_cols = ['date_time', 'magnitude']
features = [col for col in df.columns if col not in exclude_cols]

X = df[features]
y = df['magnitude']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

importances = best_model.feature_importances_
feature_names = X_train.columns
feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance')

plt.figure(figsize=(12,8))
plt.barh(feat_df['feature'], feat_df['importance'])
plt.title("Feature Importance")
plt.show()

plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Magnitude")
plt.show()

def get_new_earthquakes(min_mag=4.0):
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(minutes=10)

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": min_mag
    }

    response = requests.get(url, params=params)
    data = response.json()

    events = []
    for feature in data.get('features', []):
        p = feature['properties']
        c = feature['geometry']['coordinates']
        ts = pd.to_datetime(p['time'], unit='ms')

        event = {
            'id': feature['id'],
            'cdi': p.get('cdi', 0),
            'mmi': p.get('mmi', 0),
            'sig': p.get('sig', 0),
            'depth': c[2],
            'latitude': c[1],
            'longitude': c[0],
            'year': ts.year,
            'month': ts.month,
            'day': ts.day,
            'hour': ts.hour,
            'dayofweek': ts.dayofweek
        }
        events.append(event)

    return pd.DataFrame(events)

CHECK_INTERVAL = 600  
processed_events = set()
geolocator = Nominatim(user_agent="earthquake_monitor")

print("\n🔵 LIVE Earthquake Monitoring Started (alert only ≥ 6.0)\n")

while False:
    try:
        new_data = get_new_earthquakes(min_mag=4.0)

        if new_data.empty:
            print("No new earthquakes.")
        else:
            new_data = new_data[~new_data['id'].isin(processed_events)]

            if not new_data.empty:
                new_data = new_data.set_index("id")

                for col in X_train.columns:
                    if col not in new_data.columns:
                        new_data[col] = 0
                new_data = new_data[X_train.columns]

                for event_id, row in new_data.iterrows():
                    predicted_mag = best_model.predict(row.values.reshape(1, -1))[0]
                    processed_events.add(event_id)

                    lat = row["latitude"]
                    lon = row["longitude"]

                    try:
                        location = geolocator.reverse((lat, lon), language='en', timeout=10)
                        country = location.raw['address'].get('country', 'Unknown')
                    except:
                        country = 'Unknown'

                    if predicted_mag >= 6.0:
                        print(f"\n STRONG EARTHQUAKE ALERT ")
                        print(f"Magnitude: {predicted_mag:.2f}")
                        print(f"Location: {country} ({lat}, {lon})\n")

        time.sleep(CHECK_INTERVAL)

    except Exception as e:
        print("Error:", e)
        time.sleep(60)