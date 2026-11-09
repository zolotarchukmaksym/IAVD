from mesa import Agent, Model, DataCollector
from mesa.space import MultiGrid
import matplotlib.pyplot as plt
import numpy as np

class MoneyAgent(Agent):

    def __init__(self, model):
        super().__init__(model)
        self.wealth = 1

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
            other = self.random.choice(cellmates)
            if other.wealth > 0: # Check if other agent can receive? No, usually giver needs wealth.
                 pass
            # Standard rule: if I have money, I give 1 unit to someone else in the same cell
            if self.wealth > 0:
                other = self.random.choice(cellmates)
                other.wealth += 1
                self.wealth -= 1

    def step(self):
        self.move()
        if self.wealth > 0:
            self.give_money()

class BoltzmannWealthModel(Model):
    """
    Модель проста: агенти рухаються випадковим чином і передають гроші.
    """
    def __init__(self, N, width, height):
        super().__init__()
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)

        for i in range(self.num_agents):
            a = MoneyAgent(self)
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
    # Запуск моделі
    print("Запуск моделі багатства Больцмана...")
    model = BoltzmannWealthModel(100, 10, 10)

    for i in range(100):
        model.step()

    print("Симуляція завершена.")

    # Візуалізація розподілу багатства
    agent_wealth = [a.wealth for a in model.agents]
    plt.figure(figsize=(10, 6))
    plt.hist(agent_wealth, bins=range(max(agent_wealth)+1), edgecolor='black')
    plt.title("Розподіл багатства агентів")
    plt.xlabel("Багатство")
    plt.ylabel("Кількість агентів")
    plt.grid(True, alpha=0.3)
    plt.savefig('wealth_distribution_base.png')
    print("Графік збережено як 'wealth_distribution_base.png'")

    # Візуалізація коефіцієнта Джині
    gini = model.datacollector.get_model_vars_dataframe()
    plt.figure(figsize=(10, 6))
    plt.plot(gini)
    plt.title("Коефіцієнт Джині з часом")
    plt.xlabel("Кроки")
    plt.ylabel("Коефіцієнт Джині")
    plt.grid(True, alpha=0.3)
    plt.savefig('gini_coefficient_base.png')
    print("Графік збережено як 'gini_coefficient_base.png'")
