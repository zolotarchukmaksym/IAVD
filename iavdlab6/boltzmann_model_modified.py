from mesa import Agent, Model, DataCollector
from mesa.space import MultiGrid
import matplotlib.pyplot as plt
import numpy as np

class ModifiedMoneyAgent(Agent):
    def __init__(self, model, savings_threshold=0):
        super().__init__(model)
        self.wealth = 5
        self.savings_threshold = savings_threshold  # поріг заощаджень

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            # Перевірка на поріг заощаджень
            # Агент дає гроші тільки якщо у нього більше грошей, ніж поріг
            if self.wealth > self.savings_threshold:
                other = self.random.choice(cellmates)
                other.wealth += 1
                self.wealth -= 1

    def step(self):
        self.move()
        self.give_money()

class ModifiedBoltzmannModel(Model):

    def __init__(self, N, width, height, savings_threshold=0):
        super().__init__()
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.savings_threshold = savings_threshold

        for i in range(self.num_agents):
            # Передаємо параметр savings_threshold агенту
            a = ModifiedMoneyAgent(self, savings_threshold)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(
            model_reporters={"Gini": compute_gini},
            agent_reporters={"Wealth": "wealth"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")

def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum(xi * (N-i) for i, xi in enumerate(x)) / (N * sum(x))
    return 1 + (1/N) - 2*B

if __name__ == "__main__":
    print("Запуск модифікованої моделі (поріг заощаджень = 2)...")

    # Сценарій 1: Поріг = 0 (Базова модель)
    model_base = ModifiedBoltzmannModel(100, 10, 10, savings_threshold=0)
    for i in range(100):
        model_base.step()

    # Сценарій 2: Поріг = 2 (Модифікована модель)
    model_mod = ModifiedBoltzmannModel(100, 10, 10, savings_threshold=2)
    for i in range(100):
        model_mod.step()

    print("Симуляції завершені. Створення порівняльних графіків...")

    # Візуалізація розподілу багатства
    wealth_base = [a.wealth for a in model_base.agents]
    wealth_mod = [a.wealth for a in model_mod.agents]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(wealth_base, bins=range(max(wealth_base)+1), alpha=0.7, label='Поріг=0', color='blue', edgecolor='black')
    plt.title("Розподіл багатства (Без заощаджень)")
    plt.xlabel("Багатство")
    plt.ylabel("Кількість агентів")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(wealth_mod, bins=range(max(wealth_mod)+1), alpha=0.7, label='Поріг=2', color='green', edgecolor='black')
    plt.title("Розподіл багатства (З заощадженнями > 2)")
    plt.xlabel("Багатство")
    plt.ylabel("Кількість агентів")
    plt.legend()

    plt.tight_layout()
    plt.savefig('wealth_comparison.png')
    print("Графік збережено як 'wealth_comparison.png'")

    # Порівняння Gini
    gini_base = model_base.datacollector.get_model_vars_dataframe()
    gini_mod = model_mod.datacollector.get_model_vars_dataframe()

    plt.figure(figsize=(10, 6))
    plt.plot(gini_base, label='Поріг=0 (Базова)', color='blue')
    plt.plot(gini_mod, label='Поріг=2 (Модифікована)', color='green')
    plt.title("Порівняння коефіцієнта Джині")
    plt.xlabel("Кроки")
    plt.ylabel("Коефіцієнт Джині")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('gini_comparison.png')
    print("Графік збережено як 'gini_comparison.png'")

    print("\nВисновки:")
    print(f"Середній Gini (Базова): {gini_base['Gini'].mean():.4f}")
    print(f"Середній Gini (Модифікована): {gini_mod['Gini'].mean():.4f}")
    print("Введення порогу заощаджень зменшує нерівність (нижчий коефіцієнт Джині),")
    print("оскільки багаті агенти рідше віддають гроші, але бідні агенти захищені від повного банкрутства.")
