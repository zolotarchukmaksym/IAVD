import numpy as np
from scipy.optimize import linprog
import pandas as pd

costs = np.array([[330, 460, 510, 390, 960],  # Запоріжжя
                  [130, 160, 470, 610, 710],  # Одеса
                  [670, 650, 540, 810, 135],  # Львів
                  [0, 0, 0, 0, 0]])           # Фіктивний
supply = np.array([10, 15, 15, 5])            # Запаси: Запоріжжя, Одеса, Львів, Фіктивний
demand = np.array([8, 7, 10, 8, 12])          # Попит: Миколаїв, Арциз, Київ, Суми, Івано-Франківськ

suppliers = ['Запоріжжя', 'Одеса', 'Львів', 'Фіктивний']
consumers = ['Миколаїв', 'Арциз', 'Київ', 'Суми', 'Івано-Франківськ']

c = costs.flatten()

# Матриця обмежень рівностей (A_eq): 4 обмеження на запаси + 5 на попит = 9 обмежень
A_eq = np.zeros((9, 20))  # 4 постачальники x 5 споживачів = 20 змінних
for i in range(4):         # Обмеження по запасах (сума по рядках = запаси)
    A_eq[i, i*5:(i+1)*5] = 1
for j in range(5):         # Обмеження по попиту (сума по стовпцях = попит)
    A_eq[4+j, j::5] = 1

# Вектор правої частини обмежень (b_eq): запаси + попит
b_eq = np.hstack((supply, demand))

res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')

plan = res.x.reshape(4, 5)
df_plan = pd.DataFrame(plan, index=suppliers, columns=consumers)

print("Оптимальний план перевезень:")
print(df_plan)
print("\nЗагальна вартість:", res.fun)
print("\nТіньові ціни для обмежень (перші 4 - запаси постачальників, наступні 5 - попит споживачів):")
print(res.eqlin.marginals)
print("\nЗнижені вартості (маржинали) для всіх маршрутів (по порядку змінних x_ij, плоский вектор 20 значень):")
print(res.lower.marginals)
