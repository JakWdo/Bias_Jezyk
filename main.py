import json
import os
import time
import logging
import random
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import re
import datetime
from typing import Dict, List, Tuple, Any

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Parametry eksperymentu
PROMPTS_FILE = "integrated_prompts.json"
QUESTIONNAIRES_FILE = "questionnaires.json"
OUTPUT_DIR = "wyniki_zintegrowane"
REPEATS_PER_CONDITION = 3
INDEPENDENT_QUESTIONS = [1, 3, 5, 7, 9, 10, 13, 15, 18, 20, 22, 24, 25, 27, 29]
INTERDEPENDENT_QUESTIONS = [2, 4, 6, 8, 11, 12, 14, 16, 17, 19, 21, 23, 26, 28, 30]

# Warunki eksperymentalne
LANGUAGES = ["polski", "angielski"]
IDENTITIES = ["amerykanska", "polska", "neutralna", "japonska"]
PROMPT_STRENGTHS = ["weak", "strong"]
CONDITIONS = [f"{lang}_{identity}_{strength}"
              for lang in LANGUAGES
              for identity in IDENTITIES
              for strength in PROMPT_STRENGTHS]

MODEL = "gpt-4.1-mini"
MAX_RETRIES = 5
RETRY_DELAY = 10

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicjalizacja klienta OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("Nie znaleziono klucza API OpenAI")


def load_json_file(file_path: str) -> Dict:
    """Ładuje dane z pliku JSON."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def json_serializer(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    # Dodaj inne konwersje typów jeśli potrzebne (np. np.int64, np.float64)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def save_json_file(data: Dict, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=json_serializer)


def create_prompt(condition: str) -> str:
    """Tworzy pełny prompt dla danego warunku."""
    prompts = load_json_file(PROMPTS_FILE)
    questionnaires = load_json_file(QUESTIONNAIRES_FILE)

    language, identity, strength = condition.split('_')
    prompt_key = f"{language}_{identity}_{strength}"

    if prompt_key not in prompts:
        raise ValueError(f"Nie znaleziono promptu dla warunku: {prompt_key}")

    prompt_text = prompts[prompt_key]

    if language == "polski":
        questionnaire = questionnaires["pl"]
        instruction = questionnaires["instruction_pl"]
    else:
        questionnaire = questionnaires["en"]
        instruction = questionnaires["instruction_en"]

    return prompt_text + questionnaire + instruction


def call_openai_api(prompt: str) -> Dict[str, int]:
    """Wykonuje wywołanie API OpenAI i przetwarza odpowiedź."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000,
                timeout=60
            )

            model_response = response.choices[0].message.content

            # Wyodrębnienie JSON z odpowiedzi
            json_match = re.search(r'```json\s*(.*?)\s*```', model_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = model_response.strip()

            answers = json.loads(json_str)

            # Walidacja odpowiedzi
            if not all(str(q) in answers for q in range(1, 31)):
                raise ValueError("Niepełna odpowiedź")

            validated_answers = {}
            for q, val in answers.items():
                num_val = int(val)
                if 1 <= num_val <= 7:
                    validated_answers[q] = num_val
                else:
                    raise ValueError(f"Wartość poza zakresem dla pytania {q}")

            return validated_answers

        except Exception as e:
            logger.warning(f"Błąd (próba {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise


def calculate_scores(responses: Dict[str, int]) -> Tuple[float, float]:
    """Oblicza wyniki dla konstruktów niezależnego i współzależnego."""
    independent_score = sum(responses[str(q)] for q in INDEPENDENT_QUESTIONS) / len(INDEPENDENT_QUESTIONS)
    interdependent_score = sum(responses[str(q)] for q in INTERDEPENDENT_QUESTIONS) / len(INTERDEPENDENT_QUESTIONS)
    return independent_score, interdependent_score


def collect_data(repeats_per_condition: int = REPEATS_PER_CONDITION) -> pd.DataFrame:
    """Zbiera dane dla wszystkich warunków eksperymentalnych."""
    results = []

    # Tworzenie katalogu na dane
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_data_dir = os.path.join(OUTPUT_DIR, "raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)

    # Ścieżka dla częściowych wyników
    partial_results_path = os.path.join(OUTPUT_DIR, "partial_results.csv")

    # Wczytanie częściowych wyników jeśli istnieją
    if os.path.exists(partial_results_path):
        try:
            partial_df = pd.read_csv(partial_results_path)
            results = partial_df.to_dict('records')
            logger.info(f"Wczytano {len(results)} częściowych wyników")
        except Exception as e:
            logger.warning(f"Nie udało się wczytać częściowych wyników: {e}")

    # Obliczenie ile eksperymentów już wykonano
    condition_counts = {}
    for result in results:
        cond = result["condition"]
        condition_counts[cond] = condition_counts.get(cond, 0) + 1

    # Iteracja przez warunki
    for condition in CONDITIONS:
        completed = condition_counts.get(condition, 0)
        remaining = repeats_per_condition - completed

        if remaining <= 0:
            continue

        language, identity, strength = condition.split('_')
        logger.info(f"Warunek {condition}: pozostało {remaining} powtórzeń")

        for repeat in tqdm(range(completed + 1, repeats_per_condition + 1), desc=f"Warunek {condition}"):
            prompt = create_prompt(condition)

            try:
                # Zapisanie promptu dla pierwszego powtórzenia
                if repeat == 1:
                    prompt_filename = f"{condition}_sample_prompt.txt"
                    with open(os.path.join(raw_data_dir, prompt_filename), "w", encoding="utf-8") as f:
                        f.write(prompt)

                # Wywołanie API
                responses = call_openai_api(prompt)

                # Zapisanie wszystkich odpowiedzi
                response_filename = f"{condition}_repeat{repeat}_response.json"
                with open(os.path.join(raw_data_dir, response_filename), "w", encoding="utf-8") as f:
                    json.dump(responses, f, indent=2)

                # Obliczanie wyników
                independent_score, interdependent_score = calculate_scores(responses)

                # Zapisujemy wyniki
                result = {
                    "condition": condition,
                    "language": language,
                    "identity": identity,
                    "strength": strength,
                    "repeat": repeat,
                    "independent_score": independent_score,
                    "interdependent_score": interdependent_score,
                    "timestamp": datetime.datetime.now().isoformat()
                }

                # Dodajemy poszczególne odpowiedzi
                for question, response in responses.items():
                    result[f"q{question}"] = response

                results.append(result)

                # Zapisujemy częściowe wyniki co 10 powtórzeń
                if repeat % 10 == 0 or repeat == repeats_per_condition:
                    partial_df = pd.DataFrame(results)
                    partial_df.to_csv(partial_results_path, index=False)

                time.sleep(random.uniform(1, 3))

            except Exception as e:
                logger.error(f"Błąd dla warunku {condition}, powtórzenia {repeat}: {e}")

    # Zapisanie końcowych wyników
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(OUTPUT_DIR, "results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Zapisano dane do {csv_path}")
        return df
    else:
        return pd.DataFrame()


def perform_three_way_anova(data: pd.DataFrame, dependent_var: str) -> Dict[str, Any]:
    """Przeprowadza trzyczynnikową analizę wariancji."""
    if data.empty:
        return {"error": "Brak danych do analizy"}

    try:
        formula = f"{dependent_var} ~ C(language) + C(identity) + C(strength) + C(language):C(identity) + C(language):C(strength) + C(identity):C(strength) + C(language):C(identity):C(strength)"
        model = sm.formula.ols(formula, data=data).fit()
        anova_table = sm.stats.anova_lm(model, type=2)

        results = {
            "model_summary": {
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "f_statistic": float(model.fvalue),
                "p_value": float(model.f_pvalue)
            },
            "anova_table": {},
            "effect_sizes": {}
        }

        total_ss = anova_table["sum_sq"].sum()

        for index, row in anova_table.iterrows():
            clean_name = index.replace("C(", "").replace(")", "")
            results["anova_table"][clean_name] = {
                "sum_sq": float(row["sum_sq"]),
                "df": int(row["df"]),
                "f": float(row["F"]),
                "p": float(row["PR(>F)"]),
                "significant": float(row["PR(>F)"]) < 0.05
            }
            eta_sq = float(row["sum_sq"] / total_ss)
            results["effect_sizes"][clean_name] = {
                "eta_sq": eta_sq,
                "interpretation": "Mały efekt" if eta_sq < 0.06 else "Średni efekt" if eta_sq < 0.14 else "Duży efekt"
            }

        return results
    except Exception as e:
        logger.error(f"Błąd podczas ANOVA: {e}")
        return {"error": str(e)}


def analyze_anglocentrism(data: pd.DataFrame) -> Dict[str, Any]:
    """Przeprowadza analizę anglocentryzmu z wykorzystaniem dwóch metod."""
    if data.empty:
        return {"error": "Brak danych do analizy"}

    results = {
        "independent_score": {},
        "interdependent_score": {},
        "euclidean_analysis": {}
    }

    for strength in PROMPT_STRENGTHS:
        strength_data = data[data["strength"] == strength]

        for language in LANGUAGES:
            # Dane dla każdego warunku
            neutral_ind = strength_data[(strength_data["language"] == language) &
                                        (strength_data["identity"] == "neutralna")]["independent_score"]
            american_ind = strength_data[(strength_data["language"] == language) &
                                         (strength_data["identity"] == "amerykanska")]["independent_score"]
            polish_ind = strength_data[(strength_data["language"] == language) &
                                       (strength_data["identity"] == "polska")]["independent_score"]
            japanese_ind = strength_data[(strength_data["language"] == language) &
                                         (strength_data["identity"] == "japonska")]["independent_score"]

            neutral_int = strength_data[(strength_data["language"] == language) &
                                        (strength_data["identity"] == "neutralna")]["interdependent_score"]
            american_int = strength_data[(strength_data["language"] == language) &
                                         (strength_data["identity"] == "amerykanska")]["interdependent_score"]
            polish_int = strength_data[(strength_data["language"] == language) &
                                       (strength_data["identity"] == "polska")]["interdependent_score"]
            japanese_int = strength_data[(strength_data["language"] == language) &
                                         (strength_data["identity"] == "japonska")]["interdependent_score"]

            key = f"{language}_{strength}"

            # Analiza tradycyjna dla niezależności
            if all(len(x) > 0 for x in [neutral_ind, american_ind, polish_ind, japanese_ind]):
                result_ind = analyze_single_dimension(neutral_ind, american_ind, polish_ind, japanese_ind)
                results["independent_score"][key] = result_ind

            # Analiza tradycyjna dla współzależności
            if all(len(x) > 0 for x in [neutral_int, american_int, polish_int, japanese_int]):
                result_int = analyze_single_dimension(neutral_int, american_int, polish_int, japanese_int)
                results["interdependent_score"][key] = result_int

            # Analiza euklidesowa
            if all(len(x) > 0 for x in [neutral_ind, american_ind, polish_ind, japanese_ind,
                                        neutral_int, american_int, polish_int, japanese_int]):
                result_eucl = analyze_euclidean(
                    neutral_ind, american_ind, polish_ind, japanese_ind,
                    neutral_int, american_int, polish_int, japanese_int
                )
                results["euclidean_analysis"][key] = result_eucl

    # Ogólne wnioski
    results["overall_conclusion"] = {
        "traditional_analysis": summarize_anglocentrism(results, "traditional"),
        "euclidean_analysis": summarize_anglocentrism(results, "euclidean")
    }

    return results


def analyze_single_dimension(neutral_data, american_data, polish_data, japanese_data) -> Dict[str, Any]:
    """Analizuje anglocentryzm dla pojedynczego wymiaru."""
    neutral_mean = neutral_data.mean()
    american_mean = american_data.mean()
    polish_mean = polish_data.mean()
    japanese_mean = japanese_data.mean()

    # Odległości
    diff_american = abs(neutral_mean - american_mean)
    diff_polish = abs(neutral_mean - polish_mean)
    diff_japanese = abs(neutral_mean - japanese_mean)

    # Określenie najbliższej kultury
    diffs = [("amerykanska", diff_american), ("polska", diff_polish), ("japonska", diff_japanese)]
    closest_culture = min(diffs, key=lambda x: x[1])[0]

    # Bootstrap dla istotności
    n_bootstrap = 10000
    delta_np_na_list = []
    delta_nj_na_list = []

    for _ in range(n_bootstrap):
        boot_neutral = np.random.choice(neutral_data, size=len(neutral_data), replace=True)
        boot_american = np.random.choice(american_data, size=len(american_data), replace=True)
        boot_polish = np.random.choice(polish_data, size=len(polish_data), replace=True)
        boot_japanese = np.random.choice(japanese_data, size=len(japanese_data), replace=True)

        boot_diff_american = abs(boot_neutral.mean() - boot_american.mean())
        boot_diff_polish = abs(boot_neutral.mean() - boot_polish.mean())
        boot_diff_japanese = abs(boot_neutral.mean() - boot_japanese.mean())

        delta_np_na_list.append(boot_diff_polish - boot_diff_american)
        delta_nj_na_list.append(boot_diff_japanese - boot_diff_american)

    ci_np_na = np.percentile(delta_np_na_list, [2.5, 97.5])
    ci_nj_na = np.percentile(delta_nj_na_list, [2.5, 97.5])

    sig_vs_polish = ci_np_na[0] > 0
    sig_vs_japanese = ci_nj_na[0] > 0

    anglocentrism_significant = closest_culture == "amerykanska" and (sig_vs_polish or sig_vs_japanese)

    return {
        "neutral_mean": float(neutral_mean),
        "american_mean": float(american_mean),
        "polish_mean": float(polish_mean),
        "japanese_mean": float(japanese_mean),
        "diff_neutral_american": float(diff_american),
        "diff_neutral_polish": float(diff_polish),
        "diff_neutral_japanese": float(diff_japanese),
        "closest_to": closest_culture,
        "ci_np_na": ci_np_na.tolist(),
        "ci_nj_na": ci_nj_na.tolist(),
        "sig_closer_than_polish": sig_vs_polish,
        "sig_closer_than_japanese": sig_vs_japanese,
        "anglocentrism_present": closest_culture == "amerykanska",
        "anglocentrism_statistically_significant": anglocentrism_significant
    }


def analyze_euclidean(neutral_ind, american_ind, polish_ind, japanese_ind,
                      neutral_int, american_int, polish_int, japanese_int) -> Dict[str, Any]:
    """Analizuje anglocentryzm w przestrzeni 2D."""
    # Punkty 2D
    neutral_point = (neutral_ind.mean(), neutral_int.mean())
    american_point = (american_ind.mean(), american_int.mean())
    polish_point = (polish_ind.mean(), polish_int.mean())
    japanese_point = (japanese_ind.mean(), japanese_int.mean())

    # Odległości euklidesowe
    dist_american = np.sqrt(np.sum(np.square(np.array(neutral_point) - np.array(american_point))))
    dist_polish = np.sqrt(np.sum(np.square(np.array(neutral_point) - np.array(polish_point))))
    dist_japanese = np.sqrt(np.sum(np.square(np.array(neutral_point) - np.array(japanese_point))))

    # Określenie najbliższej kultury
    dists = [("amerykanska", dist_american), ("polska", dist_polish), ("japonska", dist_japanese)]
    closest_culture = min(dists, key=lambda x: x[1])[0]

    # Bootstrap dla istotności
    n_bootstrap = 10000
    delta_np_na_list = []
    delta_nj_na_list = []

    for _ in range(n_bootstrap):
        boot_neutral_ind = np.random.choice(neutral_ind, size=len(neutral_ind), replace=True)
        boot_american_ind = np.random.choice(american_ind, size=len(american_ind), replace=True)
        boot_polish_ind = np.random.choice(polish_ind, size=len(polish_ind), replace=True)
        boot_japanese_ind = np.random.choice(japanese_ind, size=len(japanese_ind), replace=True)

        boot_neutral_int = np.random.choice(neutral_int, size=len(neutral_int), replace=True)
        boot_american_int = np.random.choice(american_int, size=len(american_int), replace=True)
        boot_polish_int = np.random.choice(polish_int, size=len(polish_int), replace=True)
        boot_japanese_int = np.random.choice(japanese_int, size=len(japanese_int), replace=True)

        boot_neutral_point = (boot_neutral_ind.mean(), boot_neutral_int.mean())
        boot_american_point = (boot_american_ind.mean(), boot_american_int.mean())
        boot_polish_point = (boot_polish_ind.mean(), boot_polish_int.mean())
        boot_japanese_point = (boot_japanese_ind.mean(), boot_japanese_int.mean())

        boot_dist_american = np.sqrt(np.sum(np.square(np.array(boot_neutral_point) - np.array(boot_american_point))))
        boot_dist_polish = np.sqrt(np.sum(np.square(np.array(boot_neutral_point) - np.array(boot_polish_point))))
        boot_dist_japanese = np.sqrt(np.sum(np.square(np.array(boot_neutral_point) - np.array(boot_japanese_point))))

        delta_np_na_list.append(boot_dist_polish - boot_dist_american)
        delta_nj_na_list.append(boot_dist_japanese - boot_dist_american)

    ci_np_na = np.percentile(delta_np_na_list, [2.5, 97.5])
    ci_nj_na = np.percentile(delta_nj_na_list, [2.5, 97.5])

    sig_vs_polish = ci_np_na[0] > 0
    sig_vs_japanese = ci_nj_na[0] > 0

    anglocentrism_significant = closest_culture == "amerykanska" and (sig_vs_polish or sig_vs_japanese)

    return {
        "neutral_point": neutral_point,
        "american_point": american_point,
        "polish_point": polish_point,
        "japanese_point": japanese_point,
        "dist_neutral_american": float(dist_american),
        "dist_neutral_polish": float(dist_polish),
        "dist_neutral_japanese": float(dist_japanese),
        "closest_to": closest_culture,
        "ci_np_na": ci_np_na.tolist(),
        "ci_nj_na": ci_nj_na.tolist(),
        "sig_closer_than_polish": sig_vs_polish,
        "sig_closer_than_japanese": sig_vs_japanese,
        "anglocentrism_present": closest_culture == "amerykanska",
        "anglocentrism_statistically_significant": anglocentrism_significant
    }


def summarize_anglocentrism(results: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Podsumowuje wyniki analizy anglocentryzmu."""
    if method == "traditional":
        anglocentrism_count = 0
        significant_count = 0
        total_count = 0

        for construct in ["independent_score", "interdependent_score"]:
            for key, result in results.get(construct, {}).items():
                if isinstance(result, dict) and "anglocentrism_present" in result:
                    total_count += 1
                    if result["anglocentrism_present"]:
                        anglocentrism_count += 1
                    if result["anglocentrism_statistically_significant"]:
                        significant_count += 1

        return {
            "anglocentrism_present": anglocentrism_count > total_count / 2 if total_count > 0 else False,
            "anglocentrism_statistically_significant": significant_count > 0,
            "proportion_anglocentric": anglocentrism_count / total_count if total_count > 0 else 0,
            "proportion_significant": significant_count / total_count if total_count > 0 else 0
        }

    else:  # euclidean
        anglocentrism_count = 0
        significant_count = 0
        total_count = 0

        for key, result in results.get("euclidean_analysis", {}).items():
            if isinstance(result, dict) and "anglocentrism_present" in result:
                total_count += 1
                if result["anglocentrism_present"]:
                    anglocentrism_count += 1
                if result["anglocentrism_statistically_significant"]:
                    significant_count += 1

        return {
            "anglocentrism_present": anglocentrism_count > total_count / 2 if total_count > 0 else False,
            "anglocentrism_statistically_significant": significant_count > 0,
            "proportion_anglocentric": anglocentrism_count / total_count if total_count > 0 else 0,
            "proportion_significant": significant_count / total_count if total_count > 0 else 0
        }


def analyze_data(data: pd.DataFrame) -> Dict[str, Any]:
    """Przeprowadza pełną analizę danych."""
    if data.empty:
        return {"error": "Brak danych do analizy"}

    results = {
        "descriptive_stats": {},
        "anova_results": {},
        "anglocentrism_analysis": {},
        "timestamp": datetime.datetime.now().isoformat()
    }

    # Statystyki opisowe
    for construct in ["independent_score", "interdependent_score"]:
        results["descriptive_stats"][construct] = {
            "overall": {
                "mean": float(data[construct].mean()),
                "std": float(data[construct].std()),
                "min": float(data[construct].min()),
                "max": float(data[construct].max())
            }
        }

        # Statystyki dla każdego czynnika
        for factor in ["language", "identity", "strength"]:
            results["descriptive_stats"][construct][factor] = {}
            for value in data[factor].unique():
                subset = data[data[factor] == value][construct]
                results["descriptive_stats"][construct][factor][value] = {
                    "mean": float(subset.mean()),
                    "std": float(subset.std()),
                    "n": int(len(subset))
                }

    # ANOVA
    results["anova_results"]["independent_score"] = perform_three_way_anova(data, "independent_score")
    results["anova_results"]["interdependent_score"] = perform_three_way_anova(data, "interdependent_score")

    # Analiza anglocentryzmu
    results["anglocentrism_analysis"] = analyze_anglocentrism(data)

    # Zapisanie wyników
    json_path = os.path.join(OUTPUT_DIR, "analysis_results.json")
    save_json_file(results, json_path)
    logger.info(f"Zapisano wyniki analizy do {json_path}")

    return results


def run_experiment():
    """Przeprowadza eksperyment."""
    logger.info("Rozpoczęcie eksperymentu")

    # Tworzenie katalogów
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sprawdzenie plików konfiguracyjnych
    if not os.path.exists(PROMPTS_FILE):
        logger.error(f"Brak pliku {PROMPTS_FILE}")
        return {"error": f"Brak pliku {PROMPTS_FILE}"}

    if not os.path.exists(QUESTIONNAIRES_FILE):
        logger.error(f"Brak pliku {QUESTIONNAIRES_FILE}")
        return {"error": f"Brak pliku {QUESTIONNAIRES_FILE}"}

    try:
        # Zbieranie danych
        logger.info("Rozpoczęcie zbierania danych")
        data = collect_data()

        if data.empty:
            logger.error("Nie zebrano żadnych danych!")
            return {"error": "Brak danych"}

        # Analiza danych
        logger.info("Rozpoczęcie analizy danych")
        analysis_results = analyze_data(data)

        logger.info("Eksperyment zakończony")
        return {
            "data": data,
            "analysis_results": analysis_results,
            "output_dir": OUTPUT_DIR
        }

    except Exception as e:
        logger.error(f"Błąd podczas eksperymentu: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    result = run_experiment()

    if "error" not in result:
        print(f"\nEksperyment zakończony sukcesem!")
        print(f"Liczba obserwacji: {len(result['data'])}")
        print(f"Wyniki zapisane w: {result['output_dir']}")
    else:
        print(f"\nEksperyment zakończony z błędem: {result['error']}")