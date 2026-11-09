import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

COMMODITY_TO_ANALYZE = 'Wheat flour - Retail'
COMMODITY_TO_ANALYZE_VIZ = 'Apples - Retail'

MARKET_1 = 'Fayzabad'
MARKET_2 = 'Kyiv'
INITIAL_YEAR_FILTER = 2005
FALLBACK_YEAR_FILTER = 2010
MIN_SAMPLES = 10

#ФУНКЦІЇ ДЛЯ МОДЕЛЮВАННЯ ТА АНАЛІЗУ ЧУТЛИВОСТІ
def sensitivity_analysis(model, X, title, feature_idx=0, changes=[-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]):
    feature_name = X.columns[feature_idx] if isinstance(X, pd.DataFrame) else f'Feature {feature_idx}'
    X_np = X.values if isinstance(X, pd.DataFrame) else X

    base_pred = model.predict(X_np)
    changes_effect = []

    for change in changes:
        X_modified = X_np.copy()
        X_modified[:, feature_idx] = X_modified[:, feature_idx] * (1 + change)
        pred_modified = model.predict(X_modified)
        effect = np.mean((pred_modified - base_pred) / base_pred) * 100
        changes_effect.append(effect)

    plt.figure(figsize=(10, 5))
    plt.bar([f'{c * 100:.0f}%' for c in changes], changes_effect)
    plt.xlabel(f'Зміна вхідної змінної X ({feature_name}) (%)')
    plt.ylabel('Зміна прогнозованої mp_price_usd (%)')
    plt.title(title)
    plt.show()


def extended_sensitivity_analysis(model, X_scaled, X_df, title, feature_names,
                                  changes=[-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]):
    base_pred = model.predict(X_scaled)

    results = {}
    for feature_idx, feature in enumerate(feature_names):
        changes_effect = []
        for change in changes:
            X_modified = X_scaled.copy()
            X_modified[:, feature_idx] = X_modified[:, feature_idx] * (1 + change)
            pred_modified = model.predict(X_modified)
            effect = np.mean((pred_modified - base_pred) / base_pred) * 100
            changes_effect.append(effect)
        results[feature] = changes_effect

    plt.figure(figsize=(10, 5))
    for feature in results:
        plt.plot([c * 100 for c in changes], results[feature], marker='o', label=feature)
    plt.title(title)
    plt.xlabel('Зміна вхідної змінної (%)')
    plt.ylabel('Зміна прогнозованої mp_price_usd (%)')
    plt.legend()
    plt.grid()
    plt.show()


def convert_to_usd_dynamic(row):
    currency = row['cur_name']
    price = row['mp_price']

    if currency == 'USD':
        return price

    exchange_rate = row.get(currency)

    if pd.isna(exchange_rate) or exchange_rate <= 0:
        return np.nan
    else:
        return price / exchange_rate

try:
    data = pd.read_csv('global_food_prices.csv')
    rates = pd.read_csv('rates.csv')
except FileNotFoundError:
    print("Помилка: Файл 'global_food_prices.csv' або 'rates.csv' не знайдено.")
    exit()

#ПІДГОТОВКА КУРСІВ ВАЛЮТ
rates['mp_month'] = pd.to_numeric(rates['Month/Year'].str.split('/').str[0], errors='coerce')
rates['mp_year'] = pd.to_numeric(rates['Month/Year'].str.split('/').str[1], errors='coerce')
rates_clean = rates.drop(columns=['Month/Year'])
rates_agg = rates_clean.groupby(['mp_year', 'mp_month']).mean().reset_index()

# ФІЛЬТРАЦІЯ ТА ОБРОБКА ОСНОВНОГО DF
filtered_data = data[data['mp_year'] <= INITIAL_YEAR_FILTER].copy()

# Логіка коригування фільтра
if len(filtered_data) < MIN_SAMPLES:
    filtered_data = data[data['mp_year'] <= FALLBACK_YEAR_FILTER].copy()
    print(f"Розширено фільтр до mp_year <= {FALLBACK_YEAR_FILTER}. Розмір: {len(filtered_data)}")
    if len(filtered_data) < MIN_SAMPLES:
        print(f"Критична помилка: Навіть розширений фільтр повертає менше {MIN_SAMPLES} рядків.")
else:
    print(f"Розмір відфільтрованого набору даних: {len(filtered_data)}")

#ОБРОБКА ПРОПУСКІВ ТА КОДУВАННЯ ЗМІННИХ
filtered_data['adm1_name'] = filtered_data['adm1_name'].fillna(filtered_data['adm1_name'].mode()[0])
filtered_data['cm_name'] = filtered_data['cm_name'].fillna(filtered_data['cm_name'].mode()[0])
filtered_data['adm0_name_encoded'] = filtered_data['adm0_name'].astype('category').cat.codes + 1

#ОБ'ЄДНАННЯ DF З КУРСАМИ
currency_columns = filtered_data['cur_name'].unique()
rate_columns_to_merge = [c for c in currency_columns if c in rates_agg.columns] + ['mp_year', 'mp_month']
valid_rate_cols = [c for c in rate_columns_to_merge if c in rates_agg.columns]

rates_final = rates_agg[valid_rate_cols].set_index(['mp_year', 'mp_month'])
filtered_data = filtered_data.set_index(['mp_year', 'mp_month']).join(rates_final, how='left')
filtered_data = filtered_data.reset_index()

# РОЗРАХУНОК mp_price_usd
filtered_data['mp_price_usd'] = filtered_data.apply(convert_to_usd_dynamic, axis=1)

print("=" * 60)
print("1. РОЗШИРЕНИЙ СТАТИСТИЧНИЙ АНАЛІЗ")
print("=" * 60)

## 1. Розширений статистичний аналіз
stats = filtered_data.groupby('cm_name')['mp_price'].agg(['min', 'max', 'mean', 'var', lambda x: x.quantile(0.5), lambda x: x.quantile(0.75)])
stats.columns = ['min', 'max', 'mean', 'variance', 'median', 'Q3']
print("\nБазова статистика по cm_name:")
print(stats.head())

# Порівняння періодів
period_1 = filtered_data[filtered_data['mp_year'] <= 2000]
period_2 = filtered_data[(filtered_data['mp_year'] > 2000) & (filtered_data['mp_year'] <= 2005)]
stats_period_1 = period_1.groupby('cm_name')['mp_price'].agg(['mean', 'std']).add_prefix('period1_')
stats_period_2 = period_2.groupby('cm_name')['mp_price'].agg(['mean', 'std']).add_prefix('period2_')
comparison = pd.concat([stats_period_1, stats_period_2], axis=1)

print("\nПорівняння періодів:")
with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1000):
    print(comparison)

# Boxplot
plt.figure(figsize=(12, 6))
top_cm_names = filtered_data['cm_name'].value_counts().nlargest(10).index
sns.boxplot(x='cm_name', y='mp_price', data=filtered_data[filtered_data['cm_name'].isin(top_cm_names)])
plt.xticks(rotation=45, ha='right')
plt.title(f'Boxplot цін по 10 найчастіших товарах')
plt.tight_layout()
plt.show()

print("=" * 60)
print("2. ВІЗУАЛІЗАЦІЯ ТРЕНДІВ ЦІН")
print("=" * 60)

## 2. Візуалізація трендів цін
data_commodity_viz = filtered_data[filtered_data['cm_name'] == COMMODITY_TO_ANALYZE_VIZ].copy()

if not data_commodity_viz.empty:
    data_commodity_viz['log_mp_price'] = np.log1p(data_commodity_viz['mp_price'])

    # Line Plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(x='mp_year', y='mp_price', data=data_commodity_viz)
    plt.title(f'Тренд цін для {COMMODITY_TO_ANALYZE_VIZ}')
    plt.show()

    # Гістограми з різною кількістю bins
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(data_commodity_viz['mp_price'], bins=10)
    plt.title('Гістограма (10 bins)')
    plt.subplot(1, 2, 2)
    sns.histplot(data_commodity_viz['mp_price'], bins=30)
    plt.title('Гістограма (30 bins)')
    plt.tight_layout()
    plt.show()

print("=" * 60)
print("3. ОБРОБКА ПРОПУСКІВ ТА КОДУВАННЯ ЗМІННИХ")
print("=" * 60)

## Обробка пропусків та кодування змінних
print("\nПропуски в даних:")
print(filtered_data[['adm1_name', 'cm_name']].isnull().sum())

# Scatter plot для mp_year vs закодованої adm0_name
plt.figure(figsize=(10, 5))
sns.scatterplot(x='mp_year', y='adm0_name_encoded', data=filtered_data)
plt.title('Scatter plot: mp_year vs закодована adm0_name')
plt.show()

print("=" * 60)
print("4. КОРЕЛЯЦІЙНИЙ АНАЛІЗ МІЖ ЗМІННИМИ")
print("=" * 60)

## Кореляційний аналіз між змінними
corr_matrix = filtered_data[['mp_price', 'mp_year', 'mp_month']].corr(method='pearson')
print("\nКореляційна матриця:")
print(corr_matrix)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Кореляційна матриця')
plt.show()

print("=" * 60)
print("ПРОГНОЗУВАННЯ ТА ЧУТЛИВІСТЬ (ЗМІНЕНО НА USD + КРАЇНА)")
print("=" * 60)

## Прогнозування та чутливість
data_commodity = filtered_data[filtered_data['cm_name'] == COMMODITY_TO_ANALYZE].dropna(subset=['mp_price_usd']).copy()

# АГРЕГАЦІЯ ДАНИХ ДЛЯ ЗМЕНШЕННЯ ШУМУ (Середня річна ціна по країні)
aggregated_data = data_commodity.groupby(['mp_year', 'adm0_name_encoded']).agg({
    'mp_price_usd': 'mean',
    'mp_month': 'mean'
}).reset_index()

if len(aggregated_data) < MIN_SAMPLES:
    print(f"\nПопередження: Пропущено Завдання 5, 6, 9, 11. Недостатньо агрегованих даних USD для товару '{COMMODITY_TO_ANALYZE}'.")
else:
    # Завдання 5: Лінійна регресія
    X_lin_df = aggregated_data[['mp_year', 'adm0_name_encoded']].copy()
    X_lin = X_lin_df.values
    y_lin = aggregated_data['mp_price_usd'].values

    model = LinearRegression()
    model.fit(X_lin, y_lin)
    y_pred_lin = model.predict(X_lin)

    r2_lin = r2_score(y_lin, y_pred_lin)
    print(f"\nЛінійна регресія (USD + Країна, Агрегація): R²: {r2_lin:.4f}")

    # Завдання 6: Нейронна мережа
    X_mlp_df = aggregated_data[['mp_year', 'mp_month', 'adm0_name_encoded']].copy()
    y_mlp = aggregated_data['mp_price_usd'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mlp_df)

    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    mlp.fit(X_scaled, y_mlp)
    y_pred_mlp = mlp.predict(X_scaled)
    r2_mlp = r2_score(y_mlp, y_pred_mlp)
    print(f"MLP (USD + Країна, Агрегація): R²: {r2_mlp:.4f}")

    # Візуалізація прогнозів (Завдання 5/6)
    plt.figure(figsize=(10, 5))
    plt.scatter(X_lin_df['mp_year'], y_lin, label='Реальні дані (Середня річна USD)', alpha=0.5)
    plt.plot(X_lin_df['mp_year'], y_pred_lin, color='red', label='Прогноз LR')
    plt.scatter(X_mlp_df['mp_year'], y_pred_mlp, color='blue', alpha=0.2, label='Прогноз MLP')
    plt.title(f'Прогноз ціни на {COMMODITY_TO_ANALYZE} в USD (LR vs MLP) - Агреговано')
    plt.legend()
    plt.show()

    # Завдання 9: Чутливість лінійної регресії
    sensitivity_analysis(model, X_lin_df, f'Завдання 9: Чутливість Linear Regression (USD, mp_year) - Агреговано',
                         feature_idx=0, changes=[-0.2, -0.1, -0.05, 0.05, 0.1, 0.2])

    # Завдання 11: Розширена чутливість MLP
    extended_sensitivity_analysis(mlp, X_scaled, X_mlp_df,
                                  f'Завдання 11: Розширений аналіз чутливості MLP (USD) - Агреговано',
                                  feature_names=X_mlp_df.columns)

print("=" * 60)
print("7. АНАЛІЗ ВЗАЄМОЗВ'ЯЗКІВ МІЖ РИНКАМИ")
print("=" * 60)

## 7. Аналіз взаємозв’язків між ринками
market_data = filtered_data[filtered_data['mkt_name'].isin([MARKET_1, MARKET_2])].copy()

if not market_data.empty:
    # Агрегуємо середні ціни за роками для кожного ринку
    data_market1 = market_data[market_data['mkt_name'] == MARKET_1].groupby('mp_year')['mp_price'].mean().reset_index().rename(columns={'mp_price': 'mp_price_market1'})
    data_market2 = market_data[market_data['mkt_name'] == MARKET_2].groupby('mp_year')['mp_price'].mean().reset_index().rename(columns={'mp_price': 'mp_price_market2'})

    merged_markets = pd.merge(data_market1, data_market2, on='mp_year', how='inner') # inner join для спільних років

    if len(merged_markets) > 1:
        X_market = merged_markets[['mp_price_market2']].values
        y_market = merged_markets['mp_price_market1'].values
        model_market = LinearRegression()
        model_market.fit(X_market, y_market)
        r2_market = r2_score(y_market, model_market.predict(X_market))
        print(f"Регресія цін між ринками ({MARKET_1} vs {MARKET_2}): R² = {r2_market:.4f}")

        plt.figure(figsize=(10, 5))
        plt.scatter(X_market, y_market, label='Реальні дані')
        plt.plot(X_market, model_market.predict(X_market), color='red', label='Лінійна регресія')
        plt.title(f'Регресія цін між {MARKET_1} і {MARKET_2} (середні ціни по роках)')
        plt.legend()
        plt.show()
    else:
        print(
            f"Попередження: Недостатньо спільних точок даних (менше 2) для регресії між ринками ({MARKET_1}, {MARKET_2}).")
else:
    print(f"Попередження: Один із ринків ('{MARKET_1}' або '{MARKET_2}') відсутній у даних.")

print("=" * 60)
print("8. АНАЛІЗ ВПЛИВУ ВАЛЮТНИХ КУРСІВ")
print("=" * 60)

## 8. Аналіз впливу валютних курсів
data_usd_analysis = filtered_data.dropna(subset=['mp_price_usd'])
data_usd_yearly = data_usd_analysis.groupby('mp_year')['mp_price_usd'].mean().reset_index()

if len(data_usd_yearly) > 1:
    X_usd = data_usd_yearly[['mp_year']].values
    y_usd = data_usd_yearly['mp_price_usd'].values
    model_usd = LinearRegression()
    model_usd.fit(X_usd, y_usd)
    r2_usd = r2_score(y_usd, model_usd.predict(X_usd))
    print(f"Регресія середніх цін в USD vs. mp_year: R² = {r2_usd:.4f} (на основі rates.csv)")

    plt.figure(figsize=(10, 5))
    plt.scatter(X_usd, y_usd, label='Середні ціни в USD (по роках)')
    plt.plot(X_usd, model_usd.predict(X_usd), color='red', label='Лінійна регресія')
    plt.title('Регресія середніх цін в USD (з rates.csv)')
    plt.legend()
    plt.show()
else:
    print("Попередження: Недостатньо даних для аналізу в USD.")

print("=" * 60)
print("10. ПОРІВНЯЛЬНИЙ АНАЛІЗ ГРУП ТОВАРІВ")
print("=" * 60)

## 10. Порівняльний аналіз груп товарів
categories = {
    'Bread': 'Зернові', 'Rice': 'Зернові', 'Wheat': 'Зернові',
    'Wheat flour - Retail': 'Зернові', 'Maize (white) - Retail': 'Зернові',
    'Milk': 'Молочні', 'Oil': 'Молочні', 'Butter': 'Молочні',
    'Cheese - Retail': 'Молочні',
    'Yogurt - Retail': 'Молочні',
    'Apples - Retail': 'Фрукти'
}
filtered_data['category'] = filtered_data['cm_name'].map(categories)
category_data = filtered_data.dropna(subset=['category'])

category_stats = category_data.groupby('category')['mp_price'].agg(['mean', 'std'])
print("Статистика по категоріях:")
print(category_stats)

grain_data = category_data[category_data['category'] == 'Зернові'][['mp_year', 'mp_price']].groupby(
    'mp_year').mean().reset_index()
dairy_data = category_data[category_data['category'] == 'Молочні'][['mp_year', 'mp_price']].groupby(
    'mp_year').mean().reset_index()

merged_categories = pd.merge(grain_data, dairy_data, on='mp_year', suffixes=('_grain', '_dairy'))

if len(merged_categories) > 1:
    X_cat = merged_categories[['mp_price_dairy']].values
    y_cat = merged_categories['mp_price_grain'].values
    model_cat = LinearRegression()
    model_cat.fit(X_cat, y_cat)
    r2_cat = r2_score(y_cat, model_cat.predict(X_cat))
    print(f"Регресія між середніми цінами зернових та молочних: R² = {r2_cat:.4f}")

    plt.figure(figsize=(10, 5))
    plt.scatter(X_cat, y_cat, label='Реальні дані')
    plt.plot(X_cat, model_cat.predict(X_cat), color='red', label='Лінійна регресія')
    plt.title('Регресія між середніми цінами зернових та молочних продуктів')
    plt.legend()
    plt.show()
else:
    print("Попередження: Недостатньо даних для порівняння категорій 'Зернові' та 'Молочні'.")
