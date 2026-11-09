from mesa import Agent, Model, DataCollector
from mesa.space import MultiGrid
import matplotlib.pyplot as plt


class SimpleAgent(Agent):

    def __init__(self, model):
        super().__init__(model)

    def step(self):
        """
        Крок агента: випадковий рух на одну клітинку в будь-якому напрямку.
        """
        # Отримуємо можливі позиції для переміщення (сусідні клітинки)
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,  # Включаємо діагональні напрямки
            include_center=False
        )

        # Вибираємо випадкову позицію з можливих
        new_position = self.random.choice(possible_steps)

        # Переміщуємо агента на нову позицію
        self.model.grid.move_agent(self, new_position)


class SimpleModel(Model):
    def __init__(self, N, width, height):
        """
            N: кількість агентів
            width: ширина сітки
            height: висота сітки
        """
        super().__init__()
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)

        # Додавання DataCollector для збору даних про позиції агентів
        self.datacollector = DataCollector(
            agent_reporters={"Position": lambda a: (a.pos if hasattr(a, 'pos') else None)}
        )

        # Створення агентів та розміщення їх на сітці
        # В Mesa 3.x агенти автоматично додаються до self.agents (AgentSet)
        for i in range(self.num_agents):
            a = SimpleAgent(self)

            # Випадкова початкова позиція
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

    def step(self):
        """
        В Mesa 3.x використовуємо self.agents.do() для виконання кроків агентів.
        """
        # Збираємо дані перед виконанням кроку
        self.datacollector.collect(self)

        # Виконуємо крок для всіх агентів
        # Mesa 3.x: використовуємо AgentSet.do() замість scheduler.step()
        self.agents.do("step")


if __name__ == "__main__":
    print("Запуск моделі з 10 агентами на сітці 10x10...")
    model = SimpleModel(10, 10, 10)

    print("Виконання 100 кроків симуляції...")
    for i in range(100):
        model.step()

    print("Симуляція завершена. Створення візуалізації...")

    # Отримання даних про позиції агентів
    data = model.datacollector.get_agent_vars_dataframe()

    agent_ids = data.index.get_level_values('AgentID').unique()
    first_agent_id = agent_ids[0]

    positions = data.xs(first_agent_id, level="AgentID")["Position"]

    x = [pos[0] for pos in positions if pos is not None]
    y = [pos[1] for pos in positions if pos is not None]

    plt.figure(figsize=(10, 8))

    plt.plot(x, y, 'b-', alpha=0.3, label='Траєкторія', linewidth=1)
    plt.scatter(x, y, c=range(len(x)), cmap='viridis', s=50, alpha=0.6)

    plt.scatter(x[0], y[0], c='green', s=200, marker='o',
                edgecolors='black', linewidths=2, label='Початок', zorder=5)
    plt.scatter(x[-1], y[-1], c='red', s=200, marker='s',
                edgecolors='black', linewidths=2, label='Кінець', zorder=5)

    plt.xlabel("X координата", fontsize=12)
    plt.ylabel("Y координата", fontsize=12)
    plt.title(f"Траєкторія агента #{first_agent_id} після 100 кроків", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, 9.5)
    plt.ylim(-0.5, 9.5)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(x)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Крок симуляції', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig('/Users/Study/iavdlab6/agent_trajectory.png', dpi=300, bbox_inches='tight')
    print("✓ Графік траєкторії збережено як 'agent_trajectory.png'")
    plt.show()

    plt.figure(figsize=(10, 8))

    final_step = data.index.get_level_values('Step').max()
    final_positions = data.xs(final_step, level="Step")["Position"]

    final_x = [pos[0] for pos in final_positions if pos is not None]
    final_y = [pos[1] for pos in final_positions if pos is not None]

    plt.scatter(final_x, final_y, c='blue', s=300, alpha=0.6, edgecolors='black', linewidths=2)

    agent_positions = list(zip(final_positions.index, final_x, final_y))
    for agent_id, xi, yi in agent_positions:
        plt.text(xi, yi, str(agent_id), ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    plt.xlabel("X координата", fontsize=12)
    plt.ylabel("Y координата", fontsize=12)
    plt.title(f"Позиції всіх {len(final_x)} агентів на кроці {final_step}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, 9.5)
    plt.ylim(-0.5, 9.5)

    plt.tight_layout()
    plt.savefig('/Users/Study/iavdlab6/all_agents_final.png', dpi=300, bbox_inches='tight')
    print("✓ Графік фінальних позицій збережено як 'all_agents_final.png'")
    plt.show()

    print("\n" + "="*60)
    print("✓ Завдання 1 і 2 виконано успішно!")
    print("="*60)
    print(f"Створено модель з {model.num_agents} агентами")
    print(f"Виконано {final_step + 1} кроків симуляції")
    print(f"Згенеровано 2 графіки візуалізації")
    print("="*60)
