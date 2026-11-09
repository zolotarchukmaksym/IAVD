import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import linprog

# Матриця витрат C
C_matrix = np.array([
    [330, 460, 510, 390, 960],
    [130, 160, 470, 610, 710],
    [670, 650, 540, 810, 135],
    [0, 0, 0, 0, 0]
])

# Оптимальний план X_opt
X_opt = np.array([
    [0.0, 0.0, 2.0, 8.0, 0.0],
    [8.0, 7.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 3.0, 0.0, 12.0],
    [0.0, 0.0, 5.0, 0.0, 0.0]
])

# Тіньові ціни
shadow_prices = np.array([510., 310., 540., -0., -180., -150., -0., -120., -405.])

# Розділяємо та форматуємо дані
sp_suppliers = shadow_prices[:4]
sp_consumers = shadow_prices[4:]

suppliers_names = ['Запоріжжя', 'Одеса', 'Львів', 'Фіктивний']
consumers_names = ['Миколаїв', 'Арциз', 'Київ', 'Суми', 'Ів-Фр']

# Форматовані назви вузлів для графа (з Тіньовими Цінами)
suppliers = [f"{n} ({s:.0f})" for n, s in zip(suppliers_names, sp_suppliers)]
consumers = [f"{n} ({c:.0f})" for n, c in zip(consumers_names, sp_consumers)]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Теплокарта Витрат (C)
im_c = axes[0].imshow(C_matrix, cmap='viridis')
axes[0].set_title('Теплокарта Витрат (C) - Вартість', fontsize=14)
axes[0].set_xticks(np.arange(len(consumers_names)))
axes[0].set_yticks(np.arange(len(suppliers_names)))
axes[0].set_xticklabels(consumers_names)
axes[0].set_yticklabels(suppliers_names)
plt.colorbar(im_c, ax=axes[0], label='Вартість')

# Додаємо числові значення
for i in range(C_matrix.shape[0]):
    for j in range(C_matrix.shape[1]):
        axes[0].text(j, i, C_matrix[i, j], ha="center", va="center", color="w", fontsize=8)

# 2. Теплокарта Оптимального Плану (X_opt)
im_x = axes[1].imshow(X_opt, cmap='Reds')
axes[1].set_title('Теплокарта Оптимального Плану (X*) - Обсяги', fontsize=14)
axes[1].set_xticks(np.arange(len(consumers_names)))
axes[1].set_yticks(np.arange(len(suppliers_names)))
axes[1].set_xticklabels(consumers_names)
axes[1].set_yticklabels(suppliers_names)
plt.colorbar(im_x, ax=axes[1], label='Обсяг перевезень')

# Додаємо числові значення
for i in range(X_opt.shape[0]):
    for j in range(X_opt.shape[1]):
        color = "black" if X_opt[i, j] > 6 else "w"
        axes[1].text(j, i, f"{X_opt[i, j]:.1f}", ha="center", va="center", color=color, fontsize=8, fontweight='bold')

plt.tight_layout()
plt.show() # Відображення графіків

df_plan = pd.DataFrame(X_opt, index=suppliers_names, columns=consumers_names)

styled_df = df_plan.style.background_gradient(cmap='Reds', subset=pd.IndexSlice[:, :]) \
                    .set_caption("Оптимальний План Перевезень (Градієнт обсягів)") \
                    .format("{:.1f}") \
                    .set_properties(**{'font-size': '10pt'})

G = nx.DiGraph()

# 1. Створення позицій вузлів (дворядкова схема)
pos = {}
for i, s_node in enumerate(suppliers):
    pos[s_node] = (i * 2, 1) # Постачальники
for i, c_node in enumerate(consumers):
    pos[c_node] = (i * 1.5 - 1, 0) # Споживачі

# Додавання вузлів та ребер з потоками
G.add_nodes_from(suppliers)
G.add_nodes_from(consumers)

edge_labels = {}
for i in range(X_opt.shape[0]):
    for j in range(X_opt.shape[1]):
        flow = X_opt[i, j]
        if flow > 1e-6:
            u = suppliers[i]
            v = consumers[j]
            G.add_edge(u, v, weight=flow)
            edge_labels[(u, v)] = f"{flow:.1f}"

# 2. Налаштування та малювання графа
plt.figure(figsize=(18, 8))

node_colors = ['#4CAF50'] * len(suppliers) + ['#2196F3'] * len(consumers)
initial_bounds = np.array([10, 15, 15, 5, 8, 7, 10, 8, 12])
node_sizes = initial_bounds * 150

# Малювання елементів
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)
nx.draw_networkx_edges(G, pos, edgelist=G.edges, width=2.0, alpha=0.6, edge_color='gray', arrows=True)

# Підписи на ребрах (обсяги перевезень)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=10, bbox={'boxstyle':"round", 'facecolor':'white', 'alpha':0.7})

# Підписи вузлів (включають Тіньові Ціни)
node_labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_weight='bold')

plt.title('Оптимальний Транспортний Граф (Обсяги та Тіньові Ціни у Вузлах)', fontsize=16)
plt.axis('off')
plt.show() # Відображення графіків
