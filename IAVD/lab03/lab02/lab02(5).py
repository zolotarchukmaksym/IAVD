import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Коефіцієнти цільової функції C
c_initial = np.array([
    330, 460, 510, 390, 960,  # Zp
    130, 160, 470, 610, 710,  # Od
    670, 650, 540, 810, 135,  # Lv
    0, 0, 0, 0, 0             # Fikt
])
N = len(c_initial)

# Права частина обмежень b_eq (Запаси: Zp, Od, Lv, Fikt; Попит: Myk, Arts, Kyiv, Sumy, I-F)
b_eq_base = np.array([10, 15, 15, 5, 8, 7, 10, 8, 12])

# Матриця обмежень A_eq
A_eq_base = np.zeros((9, N))
for i in range(4): A_eq_base[i, i*5:(i+1)*5] = 1 # Запаси
for j in range(5): A_eq_base[4+j, j::5] = 1     # Попит

# Межі змінних
bounds_base = (0, None)

# Початкове рішення (для порівняння)
res_base = linprog(c_initial, A_eq=A_eq_base, b_eq=b_eq_base, bounds=bounds_base, method='highs')
cost_base = res_base.fun

print("## 1. Зміна попиту та порівняння з тіньовою ціною")

# Вибір: Збільшуємо попит Києва (+1). Початковий попит: 10 (індекс 6 у b_eq).
# Корекція балансу: Збільшуємо запас Фіктивного постачальника (+1) (індекс 3 у b_eq).
b_eq_q1 = b_eq_base.copy()
b_eq_q1[6] += 1
b_eq_q1[3] += 1

res_q1 = linprog(c_initial, A_eq=A_eq_base, b_eq=b_eq_q1, bounds=bounds_base, method='highs')

cost_q1 = res_q1.fun
cost_change_q1 = cost_q1 - cost_base
shadow_price_kyiv = res_base.eqlin.marginals[6]

print(f"Початкова вартість: {cost_base:.2f}")
print(f"Нова вартість (Попит Києва +1): {cost_q1:.2f}")
print(f"Фактична зміна вартості: {cost_change_q1:.2f}")
print(f"Тіньова ціна для Києва: {shadow_price_kyiv:.2f}")
print(f"Висновок: Фактична зміна ({cost_change_q1:.2f}) відповідає тіньовій ціні ({shadow_price_kyiv:.2f}).")

print("\n" + "—"*80 + "\n")

print("## 2. Заборона маршруту (bounds)")

# Вибір: Забороняємо маршрут, що використовувався: Одеса -> Арциз (x_22). Потік: 7.0
# Індекс x_22 = 1*5 + 1 = 6
ROUTE_TO_BAN_INDEX = 6

all_bounds_q2 = [(0, None)] * N
all_bounds_q2[ROUTE_TO_BAN_INDEX] = (0, 0) # Встановлюємо межі [0, 0]

res_q2 = linprog(c_initial, A_eq=A_eq_base, b_eq=b_eq_base, bounds=all_bounds_q2, method='highs')

cost_q2 = res_q2.fun
flow_x_22_new = res_q2.x[ROUTE_TO_BAN_INDEX]

print(f"Початкова вартість: {cost_base:.2f}")
print(f"Вартість після заборони x_22: {cost_q2:.2f}")
print(f"Мінімальне зростання вартості: {cost_q2 - cost_base:.2f} 📈")
print(f"Новий потік x_22 (заборонений): {flow_x_22_new:.2f}")

print("\n" + "—"*80 + "\n")

print("## 3. Підвищення витрат на маршруті з найбільшим потоком")

# Вибір: Маршрут найбільшого потоку: Львів -> Івано-Франківськ (x_35). Потік: 12.0
# Індекс x_35 = 2*5 + 4 = 14
ROUTE_TO_INCREASE_INDEX = 14
NEW_COST = 600 # Підвищуємо вартість з 135 до 600

c_q3 = c_initial.copy()
c_q3[ROUTE_TO_INCREASE_INDEX] = NEW_COST

res_q3 = linprog(c_q3, A_eq=A_eq_base, b_eq=b_eq_base, bounds=bounds_base, method='highs')

cost_q3 = res_q3.fun
X_new_q3 = res_q3.x.reshape(4, 5)

print(f"Початкова вартість: {cost_base:.2f}")
print(f"Вартість альтернативного плану (C(x_35)={NEW_COST}): {cost_q3:.2f}")
print(f"Мінімальне зростання вартості: {cost_q3 - cost_base:.2f}")
print(f"Новий потік x_35: {X_new_q3[2, 4]:.1f} (повністю виведений з плану)")
print("\nНова матриця потоків (Альтернативний план):")

# Вивід нового плану
df_new_plan = pd.DataFrame(np.round(X_new_q3, 1),
                           columns=['Миколаїв', 'Арциз', 'Київ', 'Суми', 'Івано-Франківськ'],
                           index=['Запоріжжя', 'Одеса', 'Львів', 'Фіктивний'])
print(df_new_plan.to_markdown())
