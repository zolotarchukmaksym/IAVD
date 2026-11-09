import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

OUTPUT_DIR = "results_part3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Лінгвістичні змінні
temperature = ctrl.Antecedent(np.arange(15, 41, 1), "temperature")  # °C
humidity = ctrl.Antecedent(np.arange(20, 101, 1), "humidity")  # %
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), "f_speed")  #an %

# 2. Функції належності
temperature["cold"] = fuzz.trapmf(temperature.universe, [15, 15, 18, 22])
temperature["mild"] = fuzz.gaussmf(temperature.universe, 26, 3)
temperature["hot"] = fuzz.trapmf(temperature.universe, [28, 32, 40, 40])

humidity["dry"] = fuzz.trapmf(humidity.universe, [20, 20, 30, 45])
humidity["comfy"] = fuzz.gaussmf(humidity.universe, 55, 8)
humidity["humid"] = fuzz.trimf(humidity.universe, [65, 90, 100])

fan_speed["low"] = fuzz.trimf(fan_speed.universe, [0, 0, 30])
fan_speed["medium"] = fuzz.trimf(fan_speed.universe, [20, 50, 80])
fan_speed["high"] = fuzz.trapmf(fan_speed.universe, [70, 85, 100, 100])

# 3. База правил
rule1 = ctrl.Rule(temperature["hot"] | humidity["humid"], fan_speed["high"])
rule2 = ctrl.Rule(temperature["mild"] & humidity["comfy"], fan_speed["medium"])
rule3 = ctrl.Rule(temperature["cold"] & humidity["dry"], fan_speed["low"])
rule4 = ctrl.Rule(temperature["hot"] & humidity["dry"], fan_speed["medium"])
rule5 = ctrl.Rule(temperature["mild"] & humidity["humid"], fan_speed["high"])

# 4. Створення контролера та симуляція
ac_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
ac_sim = ctrl.ControlSystemSimulation(ac_control)

ac_sim.input["temperature"] = 29
ac_sim.input["humidity"] = 70
ac_sim.compute()

print(f"Рекомендована швидкість вентилятора: {ac_sim.output['fan_speed']:.2f}%")

plt.figure()
temperature.view()
plt.title("Функції належності: температура")
plt.savefig(os.path.join(OUTPUT_DIR, "part3_temperature_mf.png"))

plt.figure()
humidity.view()
plt.title("Функції належності: вологість")
plt.savefig(os.path.join(OUTPUT_DIR, "part3_humidity_mf.png"))

plt.figure()
fan_speed.view()
plt.title("Функції належності: швидкість вентилятора")
plt.savefig(os.path.join(OUTPUT_DIR, "part3_fan_mf.png"))

plt.figure()
fan_speed.view(sim=ac_sim)
plt.title("Результат роботи системи кондиціонера")
plt.savefig(os.path.join(OUTPUT_DIR, "part3_fan_result.png"))

print(f"Графіки збережено в папку {OUTPUT_DIR}")


