import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Витрати C (4x5, Zp, Od, Lv, Fikt -> Myk, Arts, Kyiv, Sumy, I-F)
C_initial = np.array([
    [330, 460, 510, 390, 960],
    [130, 160, 470, 610, 710],
    [670, 650, 540, 810, 135],
    [0, 0, 0, 0, 0]  # Фіктивний постачальник
])
num_suppliers, num_consumers = C_initial.shape

# Початкові праві частини обмежень b_eq
b_suppliers_base = np.array([10, 15, 15, 5])
b_consumers = np.array([8, 7, 10, 8, 12])
total_demand = np.sum(b_consumers)

# Індекси для аналізу
ZAPORIZHZHYA_INDEX = 0
ZAPORIZHZHYA_STOCK_BASE = b_suppliers_base[ZAPORIZHZHYA_INDEX]


def solve_transport_problem(b_zp_new, c_base, b_suppliers_base, b_consumers):
    # Створення тимчасових даних
    C_temp = c_base.copy()
    b_suppliers_temp = b_suppliers_base.copy()
    b_suppliers_temp[ZAPORIZHZHYA_INDEX] = b_zp_new
    total_supply_current = np.sum(b_suppliers_temp)

    # 2.1. Корекція балансу (додавання/корекція Фіктивного)

    # Початковий фіктивний постачальник (останній рядок C_initial) і запас (останній елемент b_suppliers_base)
    # Ми обнуляємо початковий фіктивний запас і дозволяємо системі розрахувати новий
    b_suppliers_temp[-1] = 0
    total_supply_current = np.sum(b_suppliers_temp)

    balance_diff = total_supply_current - total_demand

    if np.isclose(balance_diff, 0):
        # Система збалансована без початкового фіктивного постачальника
        C_final = C_temp[:-1, :]  # Видаляємо рядок Фіктивного
        b_suppliers_final = b_suppliers_temp[:-1]
        b_consumers_final = b_consumers
        is_balanced = True

    elif balance_diff > 0:
        # Надлишок пропозиції (Supply > Demand). Потрібен Фіктивний Споживач.
        # Додаємо стовпець нульових витрат до C
        C_final = np.hstack((C_temp, np.zeros((C_temp.shape[0], 1))))
        b_suppliers_final = b_suppliers_temp
        # Додаємо попит для Фіктивного Споживача
        b_consumers_final = np.hstack((b_consumers, [balance_diff]))
        is_balanced = False

    else:  # balance_diff < 0
        # Надлишок попиту (Demand > Supply). Потрібен Фіктивний Постачальник.
        # Корегуємо Фіктивного Постачальника (останній рядок у C_temp, який має 0 витрат)
        b_suppliers_temp[-1] = -balance_diff  # Запас Фіктивного = дефіцит
        C_final = C_temp
        b_suppliers_final = b_suppliers_temp
        b_consumers_final = b_consumers
        is_balanced = False

    # 2.2. Формування матриці обмежень A_eq і вектора b_eq

    # Формування вектора витрат C (плоский)
    c_flat = C_final.flatten()

    # Кількість змінних
    N = C_final.size
    num_s = C_final.shape[0]
    num_c = C_final.shape[1]

    # Створення A_eq (Обмеження запасів + Обмеження попиту)
    A_eq = np.zeros((num_s + num_c, N))

    # Обмеження запасів
    for i in range(num_s):
        A_eq[i, i * num_c:(i + 1) * num_c] = 1

    # Обмеження попиту
    for j in range(num_c):
        A_eq[num_s + j, j::num_c] = 1

    # Вектор b_eq
    b_eq = np.hstack((b_suppliers_final, b_consumers_final))

    # 2.3. Розв'язання задачі
    res = linprog(c_flat, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')

    return res, total_supply_current, balance_diff, num_s, num_c

# Діапазон зміни запасу Zp: +/- 50%
percent_range = 0.50
num_steps = 20

zp_stock_range = np.linspace(
    ZAPORIZHZHYA_STOCK_BASE * (1 - percent_range),
    ZAPORIZHZHYA_STOCK_BASE * (1 + percent_range),
    num_steps
)
zp_stock_range = np.unique(np.round(zp_stock_range, 1))  # Округлення для чистоти

results = []
initial_plan = None
last_plan = None

for i, b_zp_new in enumerate(zp_stock_range):

    res, total_supply, balance_diff, num_s, num_c = solve_transport_problem(
        b_zp_new, C_initial, b_suppliers_base, b_consumers
    )

    if not res.success:
        continue

    # 3.1. Визначення складу плану
    # Змінні X (потік)
    X_matrix = res.x.reshape(num_s, num_c)
    current_plan = np.where(X_matrix > 1e-6)

    # Визначаємо початковий план для порівняння
    if i == 0:
        initial_plan = set(zip(*current_plan))
        last_plan = initial_plan

    # 3.2. Фіксація зміни плану
    plan_changed = set(zip(*current_plan)) != last_plan

    # Визначення Фіктивного елемента
    if balance_diff > 0:
        # Фіктивний Споживач (останній стовпець)
        fictive_flow = np.sum(X_matrix[:, -1])
        balance_status = f"Фіктивний C: {fictive_flow:.1f}"
    elif balance_diff < 0:
        # Фіктивний Постачальник (останній рядок)
        fictive_flow = np.sum(X_matrix[-1, :])
        balance_status = f"Фіктивний S: {fictive_flow:.1f}"
    else:
        balance_status = "Баланс ⚖️"

    results.append({
        "Запас Zp": f"{b_zp_new:.1f}",
        "Баланс": balance_status,
        "Total Supply": f"{total_supply:.1f}",
        "Загальна Вартість": f"{res.fun:.2f}",
        "Зміна Плану": "✅ ЗМІНА" if plan_changed and i > 0 else "❌ СТАБІЛЬНО"
    })

    if plan_changed and i > 0:
        last_plan = set(zip(*current_plan))

df_results = pd.DataFrame(results)
print(df_results.to_markdown(index=False))
