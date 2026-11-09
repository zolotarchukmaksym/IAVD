# Імпортування бібліотек/Означення допоміжних функцій
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression

# Відключення попереджень
warnings.filterwarnings('ignore')

# Налаштування відображення
pd.set_option("display.precision", 2)
pd.options.display.float_format = '{:.2f}'.format

# Завантаження даних із файлу *.csv
df = pd.read_csv('global_food_prices.csv')

# Вибір полів для аналізу
col = ['adm0_name', 'mkt_name', 'cm_name', 'cur_name', 'pt_name', 'um_name', 'mp_month', 'mp_year', 'mp_price']
df = df[col]

# Виведення набору даних
print(df)

# Інформація про набір даних
df.info()
print(df.describe())

# Список країн
print(df['adm0_name'].value_counts().to_frame())

# Список ринків
print(df['mkt_name'].value_counts().to_frame())

# Унікальні товари
u = list(df['cm_name'].unique())
u.sort()
print(u)

# Рядки з обмінним курсом
print(df[df['cm_name'].str.contains('Exchange rate')])

# Таблиця валют
print(pd.crosstab(df["adm0_name"], df["cur_name"]))

# Валюти по країнах (з обмінним курсом)
print((df[df['um_name']=='USD/LCU']).groupby('adm0_name')['cur_name'].first().to_frame())

# Варіанти хліба
print(df[df['cm_name'].str.contains('Bread')].groupby('cm_name')['cm_name'].first().values)

# Фільтрація базових товарів
df = df[df['cm_name'].str.contains('Bread') |
        df['cm_name'].str.contains('Milk') |
        df['cm_name'].str.contains('Meat') |
        df['cm_name'].str.contains('Fish') |
        df['cm_name'].str.contains('Rice') |
        df['cm_name'].str.contains('Water') |
        df['cm_name'].str.contains('Exchange rate')]

print(df)

# Список товарів після фільтрації
u = list(df['cm_name'].unique())
u.sort()
print(u)

# Нормалізація назв товарів
df.loc[:, 'cm_name'] = df['cm_name'].apply(lambda x: str(x)[:str(x).find(' -')] if ' -' in str(x) else str(x))
df.loc[:, 'cm_name'] = df['cm_name'].apply(lambda x: str(x)[:str(x).find(' (')] if ' (' in str(x) else str(x))

# Список після нормалізації
u = list(df['cm_name'].unique())
u.sort()
print(u)

# Словник коефіцієнтів для перерахунку в кг
d = {
    'KG': 1, 'Unit': 1, 'Pound': 0.45, '400 G': 0.4, '150 G': 0.15, '5 pcs': 1,
    '500 G': 0.5, '700 G': 0.7, '800 G': 0.8, '1.1 KG': 1.1, 'Libra': 0.45,
    '900 G': 0.9, '100 KG': 100, '46 KG': 46, 'Cuartilla': 2.88, '90 KG': 90,
    '120 KG': 120, '50 KG': 50, 'MT': 1000, '45 KG': 45, '84 KG': 84,
    '100 Pounds': 45, 'Marmite': 1, '5 KG': 5, '10 KG': 10, '1.4 KG': 1.4,
    '25 KG': 25, '2 KG': 2, 'Loaf': 1, '50 Pounds': 22.5, '380 G': 0.38,
    '500 ML': 0.5, '385 G': 0.385, '0.8 KG': 0.8, '200 ML': 0.2, '20 G': 0.02,
    '1.8 KG': 1.8, '125 G': 0.125, '20 KG': 20, '170 G': 0.17, '185 G': 0.185,
    '200 G': 0.2, '1.5 KG': 1.5, '1.2 KG': 1.2, 'Bundle': 46, '160 G': 0.16,
    'Cubic meter': 1, '1.5 L': 1.5, 'L': 1, 'USD/LCU': 1
}

# Перерахунок цін на 1 кг
df['um_koef'] = df['um_name'].map(d)
df['mp_price_un'] = df['mp_price'] / df['um_koef']

# Побудова зведеної таблиці
df_res = pd.pivot_table(df, values='mp_price',
                       index=['adm0_name', 'mp_year', 'mp_month', 'mkt_name'],
                       columns=['cm_name'], aggfunc='mean', margins=False)

# Вибір стовпців
df_res = df_res[['Exchange rate', 'Bread', 'Fish', 'Meat', 'Milk', 'Rice', 'Water']]

# Скидання індексу
df_stat = df_res.reset_index()

print(df_stat)

# Статистичний аналіз
print(df_stat["adm0_name"].value_counts(normalize=True).to_frame())

# Список країн
print(df_stat["adm0_name"].unique())

# Опис для Малі
print(df_stat[df_stat['adm0_name']=='Mali'].describe())

# Опис для України
print(df_stat[df_stat['adm0_name']=='Ukraine'].describe())

# Опис для Китаю
print(df_stat[df_stat['adm0_name']=='China'].describe())

# Опис для Індії (виправлено з Bassas da India на India)
print(df_stat[df_stat['adm0_name']=='India'].describe())

# Динаміка в Індії (виправлено на India)
d = df_stat[df_stat['adm0_name']=='India'].groupby('mp_year').mean(numeric_only=True)
print(d)
d[['Milk','Rice']].plot()
plt.show()

# Динаміка в Україні
d = df_stat[df_stat['adm0_name']=='Ukraine'].groupby('mp_year').mean(numeric_only=True)
print(d)
d[['Bread', 'Meat', 'Milk', 'Rice']].plot()
plt.show()

# Динаміка в Китаї
d = df_stat[df_stat['adm0_name']=='China'].groupby('mp_year').mean(numeric_only=True)
print(d)
d[['Rice']].plot()
plt.show()

# Середні ціни по ринках Китаю
d = df_stat[df_stat['adm0_name']=='China'].groupby(['mkt_name']).mean(numeric_only=True)
print(d)
d['Rice'].plot.bar()
plt.show()

# Кругова діаграма
d['Rice'].plot.pie()
plt.show()

# Структура ринків Китаю за роками
d = df_stat[df_stat['adm0_name']=='China']
d = d[['mkt_name', 'mp_year', 'Rice']]
d = pd.pivot_table(d, values='Rice', index=['mp_year'], columns='mkt_name', aggfunc='mean', margins=False)
print(d)
d.plot.bar()
plt.show()

# Країни з USD
usd_countries = list((df[(df['cur_name']=='USD')])['adm0_name'].unique())
print(usd_countries)

# Встановлення курсу 1 для USD країн
df_stat.loc[:, 'Exchange rate'] = df_stat.apply(lambda x: 1 if x['adm0_name'] in usd_countries else x['Exchange rate'], axis=1)

# Видалення NaN в Exchange rate
df_stat_multy = df_stat[df_stat['Exchange rate'].notna()]
print(df_stat_multy)

# Список країн з курсом
print(df_stat_multy["adm0_name"].value_counts().to_frame())
print(len(df_stat_multy["adm0_name"].value_counts()))

# Ціни на рис в USD
df_stat_multy['Rice_USD'] = df_stat_multy['Rice'] / df_stat_multy['Exchange rate']
print(df_stat_multy)

# Pivot для динаміки в USD
df_stat_multy = pd.pivot_table(df_stat_multy, values='Rice_USD', index=['mp_year'], columns='adm0_name', aggfunc='mean', margins=False)
print(df_stat_multy)
print(len(df_stat_multy.columns))

# Вибір країн для порівняння
df_stat_multy = df_stat_multy[['Democratic Republic of the Congo','Somalia', 'Tajikistan', 'Ecuador']]
df_stat_multy.plot()
plt.show()

# Лінії тренду
for c in df_stat_multy.columns:
    y = df_stat_multy[c].dropna()
    x = y.index
    model = LinearRegression()
    x = np.reshape(x, (-1, 1))
    y = np.reshape(y.values, (-1, 1))
    model.fit(x, y)
    X_pred = np.append(x, [2022, 2023, 2024])
    X_pred = np.reshape(X_pred, (-1, 1))
    trend = model.predict(X_pred)
    plt.figure()
    plt.plot(X_pred, trend, "-", label='Trend')
    plt.plot(x, y, ".", label='Real data')
    plt.xlabel('Year')
    plt.ylabel('Price in USD')
    plt.legend(title=c)
    plt.show()
