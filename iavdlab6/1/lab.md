Інтелектуальний аналіз і візуалізація даних
Лабораторна робота №6
Тема: Вивчення фреймворка Mesa для моделювання
агентів
Мета
 Ознайомитися з основними принципами та можливостями
фреймворка Mesa для моделювання агентів у Python,
 Створити просту модель та візуалізувати результати
 Запустити та модифікувати один із прикладів з офіційного сайту.
Обладнання та програмне забезпечення
1. Комп'ютер із встановленим Python 3.x
2. Інтернет-доступ для встановлення необхідних бібліотек
3. Текстовий редактор або IDE (наприклад, VS Code, PyCharm)
4. Встановлена бібліотека Mesa
Встановлення Mesa
Для встановлення Mesa скористайтесь наступною командою в терміналі
або командному рядку:
pip install mesa
Завдання 1: Створення простого моделювання
1. Створення середовища та агентів:
o Створіть новий файл simple_model.py.
o Імпортуйте необхідні модулі з Mesa:
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGridfrom mesa.datacollection import DataCollector
2. Створення агента:
o Оголосіть клас SimpleAgent, що успадковує клас Agent:
class SimpleAgent(Agent):
def __init__(self, unique_id, model):
super().__init__(unique_id, model)
def step(self):
pass
3. Створення моделі:
o Оголосіть клас SimpleModel, що успадковує клас Model:
class SimpleModel(Model):
def __init__(self, N, width, height):
self.num_agents = N
self.grid = MultiGrid(width, height,
True)
self.schedule =
RandomActivation(self)
for i in range(self.num_agents):
a = SimpleAgent(i, self)
self.schedule.add(a)
x =
self.random.randrange(self.grid.width)
y =
self.random.randrange(self.grid.height)
self.grid.place_agent(a, (x, y))
def step(self):
self.schedule.step()
4. Запуск моделі:
o Додайте код для запуску моделі:
if __name__ == "__main__":
model = SimpleModel(10, 10, 10)
for i in range(100):
model.step()Завдання 2: Візуалізація результатів
1. Встановлення бібліотеки для візуалізації:
pip install matplotlib
2. Додавання збору даних:
o В SimpleModel додайте DataCollector для збору
кількості агентів на кожному кроці:
self.datacollector = DataCollector(
agent_reporters={"Position": lambda a:
(a.pos if hasattr(a, 'pos') else None)}
)
3. Оновлення методу step:
o Зберіть дані перед виконанням кроку:
def step(self):
self.datacollector.collect(self)
self.schedule.step()
4. Візуалізація даних:
o Додайте код для виведення результатів:
import matplotlib.pyplot as plt
if __name__ == "__main__":
model = SimpleModel(10, 10, 10)
for i in range(100):
model.step()
data =
model.datacollector.get_agent_vars_dataframe(
)
positions = data.xs(0,
level="AgentID")["Position"]
x = [pos[0] for pos in positions]
y = [pos[1] for pos in positions]
plt.scatter(x, y)
plt.xlabel("X координата")
plt.ylabel("Y координата")plt.title("Позиції агентів після 100
кроків")
plt.show()
Завдання 3: Запуск і модифікація прикладу з
офіційного сайту Mesa
1. Завантаження та запуск прикладу:
o Відвідайте офіційний сайт Mesa та виберіть один з прикладів.
o Завантажте та збережіть код прикладу на свій комп'ютер.
o Запустіть приклад у своєму середовищі розробки, щоб
переконатися, що він працює.
2. Модифікація прикладу:
o Виберіть аспект моделі, який ви хочете змінити. Це може бути
поведінка агентів, параметри середовища або методи
візуалізації.
o Внесіть зміни в код та задокументуйте їх.
o Перезапустіть модель, щоб перевірити ваші модифікації.
3. Документування змін:
o Опишіть, які зміни ви внесли і чому.
o Додайте коментарі до коду, щоб пояснити логіку ваших змін.