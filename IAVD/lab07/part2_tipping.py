import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import os

# Створюємо папку для результатів, якщо не існує
output_dir = "results_part2"
os.makedirs(output_dir, exist_ok=True)

# 1. Створення лінгвістичних змінних
# Antecedent - вхідні дані, Consequent - вихідні дані
food = ctrl.Antecedent(np.arange(0, 11, 1), 'food')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

# 2. Функції належності
# Автоматичні (3 рівні: poor, average, good)
food.automf(3)
service.automf(3)

# Ручні (трикутні функції для виходу)
tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

# 3. База правил
rule1 = ctrl.Rule(food['poor'] | service['poor'], tip['low'])
rule2 = ctrl.Rule(service['average'], tip['medium'])
rule3 = ctrl.Rule(service['good'] | food['good'], tip['high'])

# 4. Створення контролера
tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

# 5. Тестування
tipping.input['food'] = 6.5
tipping.input['service'] = 9.8

# Розрахунок
tipping.compute()

print(f"Рекомендовані чайові: {tipping.output['tip']:.2f}%")

# Візуалізація результату
tip.view(sim=tipping)
plt.title('Результат розрахунку чайових')
plt.savefig(os.path.join(output_dir, 'part2_tipping_result.png'))
print(f"Графік збережено у {os.path.join(output_dir, 'part2_tipping_result.png')}")

# Візуалізація функцій належності (додатково для звіту)
plt.figure()
food.view()
plt.savefig(os.path.join(output_dir, 'part2_food_mf.png'))

plt.figure()
service.view()
plt.savefig(os.path.join(output_dir, 'part2_service_mf.png'))
