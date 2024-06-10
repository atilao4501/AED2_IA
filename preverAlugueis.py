import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Função para carregar e visualizar os dados


def load_and_preview_data(filepath, header='infer'):
    df = pd.read_csv(filepath, header=header)
    # print("Primeiras linhas do dataset:\n", df.head())
    # print("Resumo do dataset:\n", df.info())
    return df


# Carrega o conjunto de dados
file_path = r"E:\Codigos\InteligenciaArtificial\AED2\alugueis.csv"
df = load_and_preview_data(file_path)

# Define a variável alvo e as features
target = "rent amount"
categorical_features = ["city", "animal", "furniture"]
numerical_features = ["area", "rooms", "bathroom", "parking spaces", "floor",
                      "hoa", "property tax", "fire insurance"]

# Separação das features e target
X = df[categorical_features + numerical_features]
y = df[target]

# Codificação de features categóricas
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')

# Aplica o OneHotEncoder para features categóricas e concatena com o DataFrame original
encoded_categorical = encoder.fit_transform(X[categorical_features])
encoded_categorical_df = pd.DataFrame(encoded_categorical.toarray(
), columns=encoder.get_feature_names_out(categorical_features))

# Concatena as features codificadas e numéricas
X_encoded = pd.concat(
    [encoded_categorical_df, X[numerical_features].reset_index(drop=True)], axis=1)

# Normalização de features numéricas
scaler = StandardScaler()
X_encoded[numerical_features] = scaler.fit_transform(
    X_encoded[numerical_features])

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42)

# Treina o modelo KNN
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

# Avalia o desempenho do modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Erro Médio Quadrático (MSE):", mse)
print("Coeficiente de Determinação (R²):", r2)

# Carrega novos dados para previsão
new_data_path = r"E:\Codigos\InteligenciaArtificial\AED2\preverAlugueis.csv"
new_data = load_and_preview_data(new_data_path)

# Pré-processamento dos novos dados
# Codificação Categórica (se necessário)
if all(col in new_data.columns for col in categorical_features):
    encoded_new_categorical = encoder.transform(new_data[categorical_features])
    encoded_new_categorical_df = pd.DataFrame(encoded_new_categorical.toarray(
    ), columns=encoder.get_feature_names_out(categorical_features))
    new_data_encoded = pd.concat(
        [encoded_new_categorical_df, new_data[numerical_features].reset_index(drop=True)], axis=1)
else:
    raise ValueError(
        "As colunas categóricas esperadas não estão presentes nos novos dados")

# Normalizar features numéricas dos novos dados
new_data_encoded[numerical_features] = scaler.transform(
    new_data_encoded[numerical_features])

# Prediz o aluguel para os novos dados
predictions = model.predict(new_data_encoded)

# Imprime as previsões de aluguel
print("Previsões de aluguel para novos imóveis:")
print(predictions)
