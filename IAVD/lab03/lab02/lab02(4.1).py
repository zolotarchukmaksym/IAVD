import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Коефіцієнти цільової функції C (витрати, рядок за порядком x_ij: Zp, Od, Lv, Fikt)
c_initial = np.array([
    330, 460, 510, 390, 960,  # Запоріжжя
    130, 160, 470, 610, 710,  # Одеса
    670, 650, 540, 810, 135,  # Львів
    0, 0, 0, 0, 0  # Фіктивний (вартість завжди 0)
])

# Обмеження-рівності (A_eq * x = b_eq)
A_eq = np.zeros((9, 20))
# Обмеження запасів (4 постачальники)
for i in range(4):
    A_eq[i, i * 5:(i + 1) * 5] = 1
# Обмеження попиту (5 споживачів)
for j in range(5):
    A_eq[4 + j, j::5] = 1

# Права частина обмежень b_eq (Запас: 10, 15, 15, 5; Попит: 8, 7, 10, 8, 12)
b_eq = np.array([10, 15, 15, 5,
                 8, 7, 10, 8, 12])

# Межі змінних (x_ij >= 0)
bounds = (0, None)

def get_non_zero_routes(result_x):
    """Повертає індекси (0-19) маршрутів з ненульовим потоком."""
    return set(np.where(result_x > 1e-6)[0])


def get_route_name(index):
    """Конвертує індекс в ім'я маршруту (Напр., 1 -> x_12)"""
    suppliers = ['Zp', 'Od', 'Lv', 'Fikt']
    consumers = ['Myk', 'Arts', 'Kyiv', 'Sumy', 'I-F']
    s_idx = index // 5
    c_idx = index % 5
    return f"x_{s_idx + 1}{c_idx + 1} ({suppliers[s_idx]} -> {consumers[c_idx]})"


routes_to_test = {
    "x_12": 1,  # Запоріжжя -> Арциз (460)
    "x_15": 4,  # Запоріжжя -> Ів.-Фр. (960)
    "x_21": 5  # Одеса -> Миколаїв (130)
}
percent_range = 0.50  # Тестуємо від -50% до +50%
num_steps = 100
results = []

# Визначаємо початковий оптимальний план
res_initial = linprog(c_initial, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
initial_plan = get_non_zero_routes(res_initial.x)

for name, index in routes_to_test.items():
    c_base = c_initial[index]

    # Визначаємо діапазон зміни вартості
    c_min = max(0, c_base * (1 - percent_range))
    c_max = c_base * (1 + percent_range)
    cost_range = np.linspace(c_min, c_max, num_steps)

    last_plan = initial_plan

    for cost in cost_range:
        c_temp = c_initial.copy()
        c_temp[index] = cost

        res = linprog(c_temp, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if res.success:
            current_plan = get_non_zero_routes(res.x)
            plan_changed = current_plan != last_plan

            if plan_changed:

                # Фіксуємо точку зламу
                entered = current_plan - last_plan
                exited = last_plan - current_plan

                # Додаємо лише першу точку зміни для даного діапазону
                if not results or results[-1]['Маршрут'] != name or results[-1]['Нова C (злам)'] == 'N/A':
                    results.append({
                        "Маршрут": get_route_name(index),
                        "Початкова C": c_base,
                        "Нова C (злам)": f"{cost:.2f}",
                        "Зміна C (%)": f"{(cost / c_base - 1) * 100:.2f}",
                        "Зміна Плану": "✅ ЗМІНИВСЯ",
                        "Увійшли": [get_route_name(i) for i in entered],
                        "Вийшли": [get_route_name(i) for i in exited],
                        "Нова Вартість": f"{res.fun:.2f}"
                    })

                last_plan = current_plan

    # Якщо план не змінився у всьому діапазоні
    if not any(r['Маршрут'] == get_route_name(index) and r['Зміна Плану'] == "ЗМІНИВСЯ" for r in results):
        results.append({
            "Маршрут": get_route_name(index),
            "Початкова C": c_base,
            "Нова C (злам)": "N/A",
            "Зміна C (%)": "N/A",
            "Зміна Плану": "НЕ ЗМІНИВСЯ",
            "Увійшли": "N/A",
            "Вийшли": "N/A",
            "Нова Вартість": f"{res.fun:.2f} (на кінці)"
        })

df_results = pd.DataFrame(results)
print(df_results.to_markdown(index=False))
