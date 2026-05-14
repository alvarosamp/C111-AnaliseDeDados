import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = r"C:\Users\vish8\OneDrive\Documentos\GitHub\C111-AnaliseDeDados\Trabalho\archive\kc_house_data.csv"

FEATURES = [
    'sqft_living', 'grade', 'bathrooms', 'bedrooms', 'floors',
    'waterfront', 'view_qtde', 'condition', 'sqft_above', 'Years', 'lat', 'long'
]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.drop(["id", "date", "zipcode"], axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    df = df[(df['bedrooms'] > 0) & (df['bedrooms'] < 20)]
    df = df[df['bathrooms'] > 0]
    q1, q3 = df['price'].quantile(0.25), df['price'].quantile(0.75)
    iqr = q3 - q1
    df = df[(df['price'] >= q1 - 1.5 * iqr) & (df['price'] <= q3 + 1.5 * iqr)]
    df.reset_index(drop=True, inplace=True)

    # Feature engineering (alinhado ao notebook)
    df['Years'] = 2026 - df['yr_built']
    df['yrs_renovated'] = df['yr_renovated'].apply(lambda x: 0 if x == 0 else 2026 - x)
    df['renovado'] = (df['yrs_renovated'] > 0).astype(int)
    df['m2_preco'] = df['price'] / df['sqft_living']
    df['bedrooms_per_bathroom'] = df['bedrooms'] / df['bathrooms'].replace(0, np.nan)
    df['view_qtde'] = df['view']
    df['view'] = df['view'].apply(lambda x: 0 if x == 0 else 1)
    df['sqft_total'] = df['sqft_living'] + df['sqft_lot']
    df['luxo'] = (df['price'] > 500000).astype(int)
    df['condicao_baixa'] = (df['condition'] <= 2).astype(int)
    df['log_price'] = np.log1p(df['price'])

    df.drop(['yr_built', 'yr_renovated'], axis=1, inplace=True)
    return df


def train_model(df):
    X = df[FEATURES].copy()
    y = df["price"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "r2":   r2_score(y_test, y_pred),
        "mae":  mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    importances = pd.Series(
        model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    return model, scaler, metrics, importances


if __name__ == "__main__":
    print("Carregando dados...")
    df = load_data()
    print(f"Registros após tratamento: {len(df):,}")

    print("\nTreinando modelo Random Forest...")
    model, scaler, metrics, importances = train_model(df)

    print(f"\nR²:   {metrics['r2']:.4f}")
    print(f"MAE:  ${metrics['mae']:,.0f}")
    print(f"RMSE: ${metrics['rmse']:,.0f}")
    print("\nImportância das features:")
    print(importances.to_string())
