import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

OUTPUT_DIR = "results_part4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_system() -> Tuple[ctrl.ControlSystem, ctrl.Antecedent, ctrl.Antecedent, ctrl.Antecedent, ctrl.Antecedent, ctrl.Antecedent, ctrl.Consequent]:
    social_media = ctrl.Antecedent(np.arange(1, 11, 1), "social_media")
    travel = ctrl.Antecedent(np.arange(1, 11, 1), "travel")
    healthy = ctrl.Antecedent(np.arange(1, 11, 1), "healthy")
    games = ctrl.Antecedent(np.arange(1, 11, 1), "games")
    social_activity = ctrl.Antecedent(np.arange(1, 11, 1), "social_activity")
    compatibility = ctrl.Consequent(np.arange(0, 101, 1), "compatibility")

    # Функції належності для соцмереж
    social_media["low"] = fuzz.trimf(social_media.universe, [1, 1, 4])
    social_media["medium"] = fuzz.gaussmf(social_media.universe, 5.5, 1.5)
    social_media["high"] = fuzz.trapmf(social_media.universe, [7, 8, 10, 10])

    # Функції належності для подорожей
    travel["low"] = fuzz.trapmf(travel.universe, [1, 1, 3, 5])
    travel["medium"] = fuzz.gaussmf(travel.universe, 6, 1.5)
    travel["high"] = fuzz.trapmf(travel.universe, [7, 8, 10, 10])

    # Функції належності для здорового способу життя
    healthy["low"] = fuzz.trimf(healthy.universe, [1, 1, 4])
    healthy["medium"] = fuzz.gaussmf(healthy.universe, 5.5, 1.5)
    healthy["high"] = fuzz.trapmf(healthy.universe, [7, 8, 10, 10])

    # Функції належності для відеоігор
    games["low"] = fuzz.trapmf(games.universe, [1, 1, 3, 5])
    games["medium"] = fuzz.gaussmf(games.universe, 6, 1.5)
    games["high"] = fuzz.trapmf(games.universe, [7, 8, 10, 10])

    # Функції належності для соціальної активності
    social_activity["low"] = fuzz.trimf(social_activity.universe, [1, 1, 4])
    social_activity["medium"] = fuzz.gaussmf(social_activity.universe, 5.5, 1.5)
    social_activity["high"] = fuzz.trapmf(social_activity.universe, [7, 8, 10, 10])

    # Функції належності для сумісності
    compatibility["low"] = fuzz.trapmf(compatibility.universe, [0, 0, 20, 40])
    compatibility["medium"] = fuzz.gaussmf(compatibility.universe, 50, 12)
    compatibility["high"] = fuzz.trimf(compatibility.universe, [60, 85, 100])
    compatibility["perfect"] = fuzz.trapmf(compatibility.universe, [80, 90, 100, 100])

    # Правила нечіткої логіки
    rules = [
        # Ідеальна сумісність
        ctrl.Rule(travel["high"] & healthy["high"] & social_activity["high"], compatibility["perfect"]),
        ctrl.Rule(social_media["high"] & social_activity["high"] & travel["high"], compatibility["perfect"]),

        # Висока сумісність
        ctrl.Rule(travel["high"] & social_activity["high"], compatibility["high"]),
        ctrl.Rule(healthy["high"] & social_activity["medium"] & travel["medium"], compatibility["high"]),
        ctrl.Rule(social_media["high"] & games["medium"] & social_activity["high"], compatibility["high"]),

        # Низька сумісність
        ctrl.Rule(social_activity["low"] & travel["low"], compatibility["low"]),
        ctrl.Rule(healthy["low"] & social_media["low"], compatibility["low"]),
        ctrl.Rule(games["high"] & social_activity["low"], compatibility["low"]),

        # Середня сумісність
        ctrl.Rule(travel["medium"] & social_activity["medium"], compatibility["medium"]),
        ctrl.Rule(healthy["medium"] & social_media["medium"], compatibility["medium"]),
        ctrl.Rule(games["medium"] & social_activity["medium"], compatibility["medium"]),
    ]

    system = ctrl.ControlSystem(rules)
    return system, social_media, travel, healthy, games, social_activity, compatibility


def score_row(sim: ctrl.ControlSystemSimulation,
              social_media_val: float,
              travel_val: float,
              healthy_val: float,
              games_val: float,
              social_activity_val: float) -> float:
    """Повертає чітку оцінку сумісності для одного респондента."""
    sim.input["social_media"] = social_media_val
    sim.input["travel"] = travel_val
    sim.input["healthy"] = healthy_val
    sim.input["games"] = games_val
    sim.input["social_activity"] = social_activity_val
    sim.compute()
    return float(sim.output["compatibility"])


def plot_memberships(social_media, travel, healthy, games, social_activity, compatibility):
    plt.figure()
    social_media.view()
    plt.title("Social Media Activity MF")
    plt.savefig(os.path.join(OUTPUT_DIR, "part4_social_media_mf.png"))
    plt.close()

    plt.figure()
    travel.view()
    plt.title("Travel Love MF")
    plt.savefig(os.path.join(OUTPUT_DIR, "part4_travel_mf.png"))
    plt.close()

    plt.figure()
    healthy.view()
    plt.title("Healthy Lifestyle MF")
    plt.savefig(os.path.join(OUTPUT_DIR, "part4_healthy_mf.png"))
    plt.close()

    plt.figure()
    games.view()
    plt.title("Video Games Love MF")
    plt.savefig(os.path.join(OUTPUT_DIR, "part4_games_mf.png"))
    plt.close()

    plt.figure()
    social_activity.view()
    plt.title("Social Activity MF")
    plt.savefig(os.path.join(OUTPUT_DIR, "part4_social_activity_mf.png"))
    plt.close()

    plt.figure()
    compatibility.view()
    plt.title("Compatibility MF")
    plt.savefig(os.path.join(OUTPUT_DIR, "part4_compatibility_mf.png"))
    plt.close()


def main(csv_path: str):
    system, social_media, travel, healthy, games, social_activity, compatibility = build_system()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV файл '{csv_path}' не знайдено. "
            "Передайте шлях через --csv або покладіть responses.csv у папку."
        )

    df = pd.read_csv(csv_path)

    column_mapping = {
        'Ваші Ім\'я та Прізвище': 'Name',
        'Наскільки ви активні у соцмережах?': 'SocialMedia',
        'Наскільки ви любите подорожувати?': 'Travel',
        'Наскільки вам важливо підтримувати здоровий спосіб життя?': 'HealthyLifestyle',
        'Наскільки ви любите відеоігри?': 'VideoGames',
        'Як часто ви потребуєте відвідувати вечірки, зустрічі з друзями чи публічні заходи?': 'SocialActivity'
    }

    df = df.rename(columns=column_mapping)

    expected_cols = {"Name", "SocialMedia", "Travel", "HealthyLifestyle", "VideoGames", "SocialActivity"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"У CSV відсутні стовпці: {', '.join(sorted(missing))}")

    scores = []
    for _, row in df.iterrows():
        sim = ctrl.ControlSystemSimulation(system)
        score = score_row(sim,
                         row["SocialMedia"],
                         row["Travel"],
                         row["HealthyLifestyle"],
                         row["VideoGames"],
                         row["SocialActivity"])
        scores.append(score)

    df["Compatibility_Score"] = scores
    df_sorted = df.sort_values(by="Compatibility_Score", ascending=False).reset_index(drop=True)
    output_csv = os.path.join(OUTPUT_DIR, "compatibility_scored.csv")
    df_sorted.to_csv(output_csv, index=False)

    top3 = df_sorted.head(3)
    plt.figure(figsize=(8, 5))
    plt.bar(top3["Name"], top3["Compatibility_Score"], color="#6a9fb5")
    plt.title("ТОП-3 кандидатів за сумісністю")
    plt.ylabel("Compatibility Score")
    plt.xlabel("Name")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "part4_top3.png"))
    plt.close()

    if not df.empty:
        example = df.iloc[0]
        sim = ctrl.ControlSystemSimulation(system)
        score_row(sim,
                 example["SocialMedia"],
                 example["Travel"],
                 example["HealthyLifestyle"],
                 example["VideoGames"],
                 example["SocialActivity"])
        plt.figure()
        compatibility.view(sim=sim)
        plt.title(f"Дефазифікація для {example['Name']}")
        plt.savefig(os.path.join(OUTPUT_DIR, "part4_example_result.png"))
        plt.close()

    plot_memberships(social_media, travel, healthy, games, social_activity, compatibility)

    print("Готово! Результати:")
    print(df_sorted[["Name", "Compatibility_Score"]].head(10).to_string(index=False))
    print(f"\nПовна таблиця: {output_csv}")
    print(f"Графіки збережено в папку {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="samp.csv",
        help="Шлях до CSV з відповідями (за замовчуванням sample_responses.csv)",
    )
    args = parser.parse_args()
    main(args.csv)
