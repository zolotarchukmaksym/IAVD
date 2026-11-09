# --- 1. Імпорт бібліотек та налаштування ---
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
import warnings

# Відключення попереджень
warnings.filterwarnings('ignore')

# Налаштування форматування чисел
pd.options.display.precision = 2
pd.options.display.float_format = '{:.2f}'.format

# --- 2. Завантаження та попередня підготовка даних (повторення Lab 1) ---

# Завантаження та фільтрація
df = pd.read_csv('global_food_prices.csv')
col = ['adm0_name', 'mkt_name', 'cm_name', 'cur_name', 'pt_name', 'um_name', 'mp_month', 'mp_year', 'mp_price']
df = df[col]
df = df[df['cm_name'].str.contains('Bread') |
        df['cm_name'].str.contains('Milk') |
        df['cm_name'].str.contains('Meat') |
        df['cm_name'].str.contains('Fish') |
        df['cm_name'].str.contains('Rice') |
        df['cm_name'].str.contains('Water') |
        df['cm_name'].str.contains('Exchange rate')]

# Нормалізація назв товарів
df.loc[:, 'cm_name'] = df['cm_name'].apply(lambda x: str(x)[:str(x).find(' -')] )
df.loc[:, 'cm_name'] = df['cm_name'].apply(lambda x: str(x) if str(x).find(' (') == -1 else str(x)[:str(x).find(' (')] )

# Словник коефіцієнтів для перерахунку в 1 кг (залишено тільки 'Exchange rate' для наступних кроків)
d = {'KG': 1, 'Unit': 1, 'Pound': 0.45, '400 G': 0.4, '150 G': 0.15, '5 pcs': 1, '500 G': 0.5,
     '700 G': 0.7, '800 G': 0.8, '1.1 KG': 1.1, 'Libra': 0.45, '900 G': 0.9,
     '100 KG': 100, '46 KG': 46, 'Cuartilla': 2.88, '90 KG': 90, '120 KG': 120,
     '50 KG': 50, 'MT': 1000, '45 KG': 45, '84 KG': 84, '100 Pounds': 45,
     'Marmite': 1, '5 KG': 5, '10 KG': 10, '1.4 KG': 1.4, '25 KG': 25, '2 KG': 2,
     'Loaf': 1, '50 Pounds': 22.5, '380 G': 0.38, '500 ML': 0.5, '385 G': 0.385,
     '0.8 KG': 0.8, '200 ML': 0.2, '20 G': 0.02, '1.8 KG': 1.8, '125 G': 0.125,
     '20 KG': 20, '170 G': 0.17, '185 G': 0.185, '200 G': 0.2, '1.5 KG': 1.5,
     '1.2 KG': 1.2, 'Bundle': 46, '160 G': 0.16, 'Cubic meter': 1, '1.5 L': 1.5,
     'L': 1, 'USD/LCU': 1}

# Перерахунок цін на одиницю (1 кг)
df['um_koef'] = df['um_name'].map(d)
df['mp_price_un'] = df['mp_price'] / df['um_koef']

# Створення фінального DataFrame з нормованими цінами
df_res = pd.pivot_table(df, values='mp_price_un',
                       index=['adm0_name', 'mp_year', 'mp_month', 'mkt_name'],
                       columns=['cm_name'], aggfunc='mean', margins=False)
df_res = df_res[['Bread', 'Fish', 'Meat', 'Milk', 'Rice', 'Water']]

# Скидання індексів
df_stat = df_res.reset_index()
print("Базовий агрегований DataFrame (df_stat, ціна за 1 кг/л):")
print(df_stat.head())

# --- 3. Функції для аналізу ---

def my_headmap(corr):
    '''
    Візуалізація кореляційної матриці.
    '''
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, cmap='RdYlGn', vmin=-1., vmax=1., annot=True, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

def my_sens(x, y, p):
    '''
    Розрахунок чутливості Y до зміни X на p відсотків.
    '''
    regressor = LinearRegression()
    regressor.fit(x, y)
    y_pred = regressor.predict(x.iloc[-1:])
    y_pred_delta = regressor.predict(x.iloc[-1:] * (1 + p))
    return ((y_pred_delta - y_pred) / y_pred)[0]

# --- 4. Завдання 1: Аналіз цін на товари в Україні ---

print("\n" + "="*50)
print("ЗАВДАННЯ 1: АНАЛІЗ ЦІН НА ТОВАРИ В УКРАЇНІ")
print("="*50)

# Створення набору даних для України
df_ukr = df_stat[df_stat['adm0_name']=='Ukraine'].groupby(['mp_year','mp_month']).mean(numeric_only=True)
df_ukr.columns.name = "Type of Purchase"
uk_col = ['Bread', 'Meat', 'Milk' ,'Rice']
df_ukr = df_ukr[uk_col]

# Візуалізація цін
df_ukr.plot(xlabel='Years, Month', ylabel='Price UAH/kg', title='Purchases in Ukraine (UAH/kg)', rot=90)
plt.show()

# Кореляційний аналіз
corr_ukr = df_ukr.corr()
print("\nКореляційна матриця (Україна):")
print(corr_ukr)
my_headmap(corr_ukr)

# Розділення на навчальний та тестовий набори (прогноз на 2017 рік)
train = df_ukr[df_ukr.index.get_level_values(0) != 2017]
test = df_ukr[df_ukr.index.get_level_values(0) == 2017]

x_col = ['Bread', 'Milk', 'Rice']
y_col = ['Meat']

x_train = train[x_col]
y_train = train[y_col]
x_test = test[x_col]
y_test = test[y_col]

# Лінійна регресія (прогноз ціни на М'ясо)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred_test = regressor.predict(x_test)
y_pred_train = regressor.predict(x_train)

# Оцінка моделі
print("\nОцінка лінійної моделі (Прогноз ціни М'яса):")
print(f"Correlation train: {regressor.score(x_train, y_train):.2f}")
print(f"Correlation test: {regressor.score(x_test, y_test):.2f}")
print("Coefficients:", regressor.coef_)
print('Intercept', regressor.intercept_)
print('Root Mean Squared Error (test):', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))

# Візуалізація результатів прогнозу
for_test = pd.DataFrame({'Actual_test': y_test.values.flatten(), 'Predicted_test': y_pred_test.flatten()})
for_test.index = y_test.index
for_train = pd.DataFrame({'Actual_train': y_train.values.flatten(), 'Predicted_train': y_pred_train.flatten()})
for_train.index = y_train.index
res = for_train.join(for_test, how='outer')
print("\nРезультати прогнозу:")
print(res)
res.plot(xlabel='Years', ylabel='Price UAH/kg', title=y_col[0] + ' Price Forecast (Linear Regression)', rot=90)
plt.show()

# Аналіз чутливості (М'ясо від Хліба)
sens_meat_to_bread = my_sens(df_ukr[['Bread']], df_ukr['Meat'], 0.1) * 100
print(f"\nSensitivity of Meat price on Bread price (10% change): {sens_meat_to_bread:.2f}%")

# Аналіз чутливості (повна матриця)
s = np.zeros((len(df_ukr.columns), len(df_ukr.columns)))
for i, r in enumerate(df_ukr.columns):
    for j, c in enumerate(df_ukr.columns):
        s[i, j] = my_sens(df_ukr[[c]], df_ukr[r], 0.1)

s = pd.DataFrame(s) * 100
s.index = df_ukr.columns.copy()
s.columns = df_ukr.columns.copy()
s.columns.name = 'input'
s.index.name = 'output'
print("\nМатриця чутливості (%) до 10% зміни вхідної ціни:")
print(s)


# --- 5. Завдання 2: Прогноз цін для китайських ринків ---

print("\n" + "="*50)
print("ЗАВДАННЯ 2: ПРОГНОЗ ЦІН ДЛЯ КИТАЙСЬКИХ РИНКІВ (РИС)")
print("="*50)

# Створення набору даних для ринків рису в Китаї
df_for = pd.pivot_table(df[(df['adm0_name'] == 'China') & (df['cm_name'] == 'Rice')],
                        values= 'mp_price_un', index= ['adm0_name', 'mp_year', 'mp_month'],
                        columns=['mkt_name'], aggfunc='mean', margins=False)
print("\nDataFrame цін на рис по ринках Китаю:")
print(df_for.head())

# Кореляційний аналіз
print("\nКореляційна матриця (Ринки Китаю, Рис):")
my_headmap(df_for.corr())

# Розділення на навчальний та тестовий набори (прогноз на 2020 рік)
train = df_for[df_for.index.get_level_values(1) != 2020]
test = df_for[df_for.index.get_level_values(1) == 2020]

x_col = ['Hangzhou', 'Jiujiang', 'Wuhan', 'Yuncheng']
y_col = ['Hubei']

# Видалення NaN, оскільки в даних є пропуски
x_train = train[x_col].dropna()
y_train = train[y_col].loc[x_train.index]
x_test = test[x_col].dropna()
y_test = test[y_col].loc[x_test.index]

# Лінійна регресія
regressor_lin = LinearRegression()
regressor_lin.fit(x_train, y_train)
y_pred_test_lin = regressor_lin.predict(x_test)
y_pred_train_lin = regressor_lin.predict(x_train)

print("\nОцінка ЛІНІЙНОЇ моделі (Прогноз ціни Hubei):")
print(f"Correlation test (Linear): {regressor_lin.score(x_test, y_test):.2f}")

# Багатошарова нейронна мережа (MLPRegressor)
regressor_mlp = MLPRegressor(hidden_layer_sizes = (10, 10), solver='lbfgs', activation = 'relu',
                          learning_rate_init=0.00001)
regressor_mlp.fit(x_train, y_train)
y_pred_test_MLP = regressor_mlp.predict(x_test)
y_pred_train_MLP = regressor_mlp.predict(x_train)

print("\nОцінка МЛП моделі:")
print(f"Correlation test (MLP): {regressor_mlp.score(x_test, y_test):.2f}")

# Візуалізація порівняння моделей
for_test = pd.DataFrame({'Actual_test': y_test.values.flatten(),
                         'Predicted_test_Lin': y_pred_test_lin.flatten(),
                         'Predicted_test_MLP': y_pred_test_MLP.flatten(),})
for_test.index = y_test.index

res_for = for_test[['Actual_test', 'Predicted_test_Lin', 'Predicted_test_MLP']]
res_for.plot(xlabel='Years, Month', ylabel='Rice price (CNY/kg)', title=y_col[0] + ' Price Forecast Comparison', rot=90)
plt.show()


# --- 6. Завдання 3: Прогноз цін в Україні від країн-експортерів ---

print("\n" + "="*50)
print("ЗАВДАННЯ 3: ПРОГНОЗ ЦІН В УКРАЇНІ ВІД ЕКСКПОРТЕРІВ")
print("="*50)

# Країни-експортери
export_counties = ['Thailand', 'Viet Nam', 'China', 'Bassas da India', 'Pakistan', 'Ukraine']

# Трансформація DataSet для задачі експорту (лише ціни на рис)
df_exp = pd.pivot_table(df[df['cm_name'] == 'Rice'],
                        values= 'mp_price_un', index= ['mp_year', 'mp_month'],
                        columns=['adm0_name'], aggfunc='mean', margins=False)
df_exp = df_exp[export_counties].dropna()

# Завантаження та підготовка курсів валют
rt = pd.read_csv('rates.csv')
rt['Month'] = pd.to_numeric(rt['Month/Year'].apply(lambda x: str(x)[: str(x).find('/')] if str(x).find("/") != -1 else ''))
rt['Year'] = pd.to_numeric(rt['Month/Year'].apply(lambda x: str(x)[str(x).find('/')+1:] if str(x).find("/") != -1 else ''))

# Визначення валют експортерів
cl = df.groupby('adm0_name')['cur_name'].first()
Filter_df = cl[cl.index.isin(export_counties)]

# Агрегація курсів валют
rt = rt[list(Filter_df.values) + ['Month', 'Year']].groupby(['Year', 'Month']).mean()

# Об'єднання цін та курсів валют
rt.index.names = df_exp.index.names
df_exp = df_exp.join(rt, how='left').dropna()

# Конвертація цін в USD
for c in df_exp.columns[:-6]: # Ціни в місцевій валюті
    df_exp[c + ' USD'] = df_exp[c] / df_exp[Filter_df[c]]

# Залишаємо тільки ціни в USD
df_exp = df_exp[df_exp.columns[-6:]]
print("\nDataFrame цін на рис в USD:")
print(df_exp.head())

# Кореляційний аналіз
print("\nКореляційна матриця (Експортери -> Україна, USD):")
my_headmap(df_exp.corr())

# Розділення на навчальний та тестовий набори (прогноз на 2017 рік)
train = df_exp[df_exp.index.get_level_values(0) != 2017]
test = df_exp[df_exp.index.get_level_values(0) == 2017]

x_col = df_exp.columns[:-1]
y_col = df_exp.columns[-1:]

x_train = train[x_col]
y_train = train[y_col]
x_test = test[x_col]
y_test = test[y_col]

# Лінійна регресія
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred_test = regressor.predict(x_test)

# Оцінка моделі
print("\nОцінка лінійної моделі (Прогноз ціни Рису в Україні від Експортерів):")
print(f"Correlation test: {regressor.score(x_test, y_test):.2f}")
print('Root Mean Squared Error (test):', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))

# Візуалізація
for_test = pd.DataFrame({'Actual_test': y_test.values.flatten(), 'Predicted_test': y_pred_test.flatten()})
for_test.index = y_test.index

res_exp = for_test[['Actual_test', 'Predicted_test']]
res_exp.plot(xlabel='Years, Month', ylabel='Rice price (USD/kg)', title='Ukraine Rice Price Forecast (Export Factors)', rot=90)
plt.show()

# Аналіз чутливості
print("\nАналіз чутливості (10% зростання цін у країнах-експортерах):")
for c in df_exp.columns[:-1]:
    sens_val = my_sens(df_exp[[c]], df_exp['Ukraine USD'], 0.1) * 100
    print(f"Sensitivity of Ukraine Rice price on {c} Rice price: {sens_val:.2f}%")
