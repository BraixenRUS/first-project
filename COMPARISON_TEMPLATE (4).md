# -*- coding: utf-8 -*-
"""
TEMPLATE: Platform Comparison Report

Каждый студент должен заполнить этот файл после работы с двумя-тремя
vibe-coding платформами. Сравните их по заданным критериям.

Инструкции:
1. Выберите 3 платформы из доступных бесплатно в РФ
 - GigaCode
 - Чаты
 - Спец-платформа для кода
2. Для каждой платформы выполните своё практическое задание
3. Заполните секции ниже с объективной оценкой
4. Приложите скриншоты и код, сгенерированный платформами
"""

# ============================================================================
# СТУДЕНТ: Тузиков Данил Максимович
# ЗАДАНИЕ: 9: GLUE MRPC Similarity
# ДАТА: 17.11.2025
# ============================================================================

## Выбранные платформы для сравнения

Укажите 3 платформы:

1. **Платформа 1:** DeepSeek
   - URL/Установка: https://chat.deepseek.com
   - Модель AI: DeepSeek AI
   - Бесплатный лимит: Полностью бесплатен без ограничений по количеству запросов.
					   Доступны все функции, включая загрузку файлов (изображений, PDF, Word, Excel, PowerPoint, TXT) 
					   и поиск в интернете (требует ручной активации)

2. **Платформа 2:** GigaCode
   - URL/Установка: Использовал в GigaIDE, подключил аккаунт из GitVerse
   - Модель AI: GigaCode
   - Бесплатный лимит: Нет ограничений по количеству запросов

3. **Платформа 3 (опционально):** Cursor
   - URL/Установка: https://cursor.com/home?from=agents скачал среду от курсора тут
   - Модель AI: Использует различные модели, включая GPT-4, GPT-3.5-Turbo и собственные модели
   - Бесплатный лимит: 7 дней или +-50 запросов (грустно)

---

## Критерии сравнения

### 1. Простота использования (1-5)
| Платформа | Оценка | Комментарий |
|-----------|--------|-----------|
| DeepSeek  | 5 | Зашёл на сайт и попросил что тебе надо без заморочек |
| GigaCode  | 3 | Пытался найти версию без IDE не нашёл, пришлось скачать. По итогу во время первого использования он не работал. Через пару дней зашёл - заработал, о чудо |
| Cursor  | 4 | Можно использовать и онлайн, но в IDE намного приятнее. Вот бы без ограничений работал красавец |

**Итоговое наблюдение:** DeepSeek

---

### 2. Качество генерируемого кода (1-5)
| Платформа | Оценка | Плюсы | Минусы |
|-----------|--------|-------|--------|
| DeepSeek | 4 | Код рабочий, с пояснениями | Бывают ошибки |
| GigaCode | 4 | Код рабочий, с пояснениями, выглядит красивенько  | Гененрировался бы когда нужно, я был бы более милосерден к нему. Код хорош, но не лучше курсора, поэтому 4 |
| Cursor | 5 | Отличный рабочий код, соответствует каждому моего требованию, придерживается предыдущих критериев отнсительно стандартов. Its briliant | Из минусов вижу только ограничение бесплатной версии |

**Итоговое наблюдение:** Cursor

---

### 3. Скорость генерации (1-5)
| Платформа | Оценка | Время первого ответа | Время генерации 50 строк |
|-----------|--------|----------------------|------------------------|
| DeepSeek    |3 | 2:25 | 0:26 |
| GigaCode | 5 | 0:41 | 0:15 |
| Cursor | 4 | 2:00 |0:20 |

**Итоговое наблюдение:** GigaCode
---

### 4. Понимание контекста (1-5)
| Платформа | Оценка | Примеры хорошего понимания | Ошибки в понимании |
|-----------|--------|---------------------------|-------------------|
| DeepSeek | 3 | В основном понимает нормально | прошу сделать единый ридми, но мне пишет отрывками или вообще просто текстом. То есть не выполняет поставленную задачу  |
Хоть это не происходило не в рамках этого задания, но если бы я попросил и тут сделать .md, то произошло бы тоже самое
```
а можно ли это написать единым md файлом на markdown, чтобы мне было легче его перенести в гитхаб?
```
```
Спецификация системы: Калькулятор математических выражений
Роль системы
Реализация консольного калькулятора на C#, который парсит и вычисляет математические выражения в инфиксной нотации с использованием стека, соблюдая приоритет операций и корректно обрабатывая скобки.

Технические требования
Ограничения
Обязательное использование структуры данных Stack для вычислений

Запрещено использовать встроенные eval-функции или библиотеки для вычисления выражений

Поддержка целых и вещественных чисел

Обработка ошибок валидации и деления на ноль
```
Выдаёт обычный текст, а не на markdown, даже после повторных просьб.
| GigaCode | 4 | Делает всё как надо | Иногда приходится вносить небольшие правки (например чтобы код подходил под стандарты, которые я уже кидал ему) |
| Cursor | 5 | Кидаю нужны стандарты, запросы, критерии - всё выполняет как надо | На моём опыте не было |

**Итоговое наблюдение:** Cursor

---

### 5. Интеграция с инструментами (1-5)
| Платформа | IDE поддержка | Git интеграция | Тестирование | Отладка |
|-----------|---------------|----------------|--------------|---------|
| DeepSeek | 3 | 2 | 3 | 3 |
| GigaCode | 4 | 4 | 4 | 4 |
| Cursor | 5 | 5 | 4 | 5 |

**Итоговое наблюдение:** Cursor

---

### 6. Бюджет и доступность (1-5)
| Платформа | Доступность в РФ | Бесплатный план | Пробный период | Стоимость |
|-----------|------------------|-----------------|----------------|-----------|
| DeepSeek | 5 | 4 | 5 | 0 |
| GigaCode | 5 | 5 | 5 | 0 |
| Cursor | 3 | 3 | 4 | 20$/месяц |

**Итоговое наблюдение:** GigaCode

---

## Финальное сравнение

### Матрица оценок (сумма по всем критериям)
| Платформа | Общая оценка (max 30) | Рекомендация |
|-----------|----------------------|--------------|
| DeepSeek | 22.4 | Хотелось бы видеть интеграцию как у Курсора, мне понравилась эта тема(думаю по моим восторженным отзывам о курсоре заметно) |
| GigaCode | 25 | Крутая тема, буду щупать в будущем. А так, мне кажется, гигакоду нужно лишь время, чтобы стоять увренно с нейро-мастадонтами |
| Cursor | 26.1 | Нужно больше бесплатного периода и запросов :) |

---

### Заключение

**Лучшая платформа для моего задания:** Cursor

**Причины выбора:**
- Наибольшее количество баллов
- Наиудобнейшее использование в среде и просто это конфетка, а не нейронка
- Крутой код, классное понимание. Меня так не понимали даже собственные родители.

**Где эта платформа может улучшиться:**
- Мне нужен бесплатный доступ
- Думаю стоит просто продолжать совршенствовать её возможности, каких-то прям идей улучшения в голову не приходит

**Другие интересные находки:**
Я думал пробный доступ на 7 дней будет работать реально 7 дней, а не полдня из-за завала моими запросами((
На удивление, система обнаружила, что я пытаюсь входить с разных акков ради бесплатного периода и приструнила меня.

---


## Рекомендации для других студентов

На основе своего опыта, рекомендую:

1. **Для быстрого прототипирования:** используйте DeepSeek, если установлен GigaIDE - GigaCode
2. **Для качественного кода:** используйте Cursor, однозначно
3. **Для обучения (понимание AI-процесса):** Интересно наблюдать за глубоким мышлением DeepSeek, стоит попробовать

## Код
- DeepSeek

```
"""
Анализ датасета GLUE MRPC (Paraphrase Detection)
"""

import json
import re
from collections import Counter
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt


def load_glue_mrpc_dataset():
    """Загружает датасет GLUE MRPC"""
    return load_dataset("glue", "mrpc")


def preprocess_text(text):
    """Токенизация текста"""
    return re.findall(r'\b\w+\b', text.lower()) if text else []


def analyze_similarity_distribution(dataset):
    """Анализ распределения парафраз/не-парафраз"""
    distribution = {}
    for split_name, split_data in dataset.items():
        labels = split_data['label']
        total = len(labels)
        paraphrase_count = sum(labels)
        distribution[split_name] = {
            'total': total,
            'paraphrase_count': paraphrase_count,
            'non_paraphrase_count': total - paraphrase_count,
            'paraphrase_percentage': round((paraphrase_count / total) * 100, 2)
        }
    return distribution


def analyze_sentence_lengths(dataset):
    """Статистика длин предложений"""
    length_stats = {}
    for split_name, split_data in dataset.items():
        sent1_lengths = [len(preprocess_text(sent)) for sent in split_data['sentence1']]
        sent2_lengths = [len(preprocess_text(sent)) for sent in split_data['sentence2']]
        length_stats[split_name] = {
            'sentence1': {
                'mean': round(np.mean(sent1_lengths), 2),
                'median': int(np.median(sent1_lengths)),
                'min': min(sent1_lengths),
                'max': max(sent1_lengths)
            },
            'sentence2': {
                'mean': round(np.mean(sent2_lengths), 2),
                'median': int(np.median(sent2_lengths)),
                'min': min(sent2_lengths),
                'max': max(sent2_lengths)
            }
        }
    return length_stats


def calculate_word_overlap(dataset):
    """Анализ пересечения слов (Jaccard similarity)"""
    overlap_stats = {}
    for split_name, split_data in dataset.items():
        similarities = []
        for sent1, sent2 in zip(split_data['sentence1'], split_data['sentence2']):
            tokens1, tokens2 = set(preprocess_text(sent1)), set(preprocess_text(sent2))
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            similarity = intersection / union if union > 0 else 0
            similarities.append(similarity)
        
        overlap_stats[split_name] = {
            'mean_similarity': round(np.mean(similarities), 3),
            'median_similarity': round(np.median(similarities), 3),
            'max_similarity': round(max(similarities), 3),
            'min_similarity': round(min(similarities), 3)
        }
    return overlap_stats


def create_visualization(dataset, filename="mrpc_distribution.png"):
    """Создает визуализацию распределения"""
    train_labels = dataset['train']['label']
    val_labels = dataset['validation']['label']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Распределение для train
    train_counts = [sum(train_labels), len(train_labels) - sum(train_labels)]
    ax1.bar(['Paraphrases', 'Non-paraphrases'], train_counts, color=['skyblue', 'lightcoral'])
    ax1.set_title('Train Set Distribution')
    ax1.set_ylabel('Count')
    
    # Распределение для validation
    val_counts = [sum(val_labels), len(val_labels) - sum(val_labels)]
    ax2.bar(['Paraphrases', 'Non-paraphrases'], val_counts, color=['skyblue', 'lightcoral'])
    ax2.set_title('Validation Set Distribution')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    """Основная функция анализа"""
    print("Загрузка датасета MRPC...")
    dataset = load_glue_mrpc_dataset()
    
    print("Анализ распределения...")
    distribution = analyze_similarity_distribution(dataset)
    
    print("Анализ длин предложений...")
    length_stats = analyze_sentence_lengths(dataset)
    
    print("Анализ пересечения слов...")
    overlap_stats = calculate_word_overlap(dataset)
    
    print("Создание визуализации...")
    create_visualization(dataset)
    
    # Сохранение результатов
    results = {
        'distribution': distribution,
        'length_stats': length_stats,
        'overlap_stats': overlap_stats
    }
    
    with open('mrpc_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Результаты сохранены в mrpc_results.json")
    print("Визуализация сохранена в mrpc_distribution.png")
    
    return results


if __name__ == "__main__":
    main()
```

- Cursor
```
"""Компактные утилиты для анализа GLUE MRPC."""

import json
import re
from typing import Dict, Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

Dataset = Iterable[Dict[str, Union[str, int]]]
VizInput = Union[Dataset, Tuple[List[int], List[int], Dict[str, int]]]
_TOKEN_RE = re.compile(r"[^a-z0-9\s]+")


def load_glue_mrpc_dataset(split: str = "train") -> Dataset:
    """Возвращает указанный сплит MRPC и печатает его размер."""
    data = load_dataset("glue", "mrpc")[split]
    print(f"Загружено {len(data)} записей для сплита {split!r}")
    return data


def preprocess_text(text: str) -> List[str]:
    """Переводит текст в нижний регистр, очищает и токенизирует."""
    if not text:
        return []
    cleaned = _TOKEN_RE.sub(" ", text.lower())
    return [token for token in cleaned.split() if token]


def _describe(lengths: List[int]) -> Dict[str, float]:
    """Возвращает сводные статистики для списка длин."""
    arr = np.array(lengths or [0], dtype=float)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
    }


def _jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Считает пересечение Жаккара для двух наборов токенов."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def analyze_dataset(dataset: Dataset):
    """Собирает распределение классов, длины и перекрытие слов."""
    len1: List[int] = []
    len2: List[int] = []
    overlaps: List[float] = []
    paraphrase = 0
    total = 0
    for row in dataset:
        tokens1 = preprocess_text(str(row["sentence1"]))
        tokens2 = preprocess_text(str(row["sentence2"]))
        len1.append(len(tokens1))
        len2.append(len(tokens2))
        overlaps.append(_jaccard(tokens1, tokens2))
        paraphrase += int(row["label"])
        total += 1
    counts = {
        "paraphrase": paraphrase,
        "non_paraphrase": total - paraphrase,
    }
    factor = 100.0 / total if total else 0.0
    similarity = {
        "counts": counts,
        "percentages": {k: v * factor for k, v in counts.items()},
    }
    lengths = {
        "sentence1": _describe(len1),
        "sentence2": _describe(len2),
    }
    arr = np.array(overlaps or [0.0], dtype=float)
    overlap = {
        "avg": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }
    return similarity, lengths, overlap, len1, len2


def analyze_similarity_distribution(dataset: Dataset) -> dict:
    """Возвращает распределение классов paraphrase/non-paraphrase."""
    return analyze_dataset(dataset)[0]


def analyze_sentence_lengths(dataset: Dataset) -> dict:
    """Возвращает статистики длины предложений."""
    return analyze_dataset(dataset)[1]


def calculate_word_overlap(dataset: Dataset) -> dict:
    """Возвращает метрики пересечения слов (Жаккар)."""
    return analyze_dataset(dataset)[2]


def create_visualization(
    data: VizInput,
    output_path: str = "visualization.png",
) -> str:
    """Строит графики распределения классов и длин предложений."""
    if isinstance(data, tuple):
        len1, len2, counts = data
    else:
        similarity, _, _, len1, len2 = analyze_dataset(data)
        counts = similarity["counts"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(
        ("Paraphrase", "Non-paraphrase"),
        [counts["paraphrase"], counts["non_paraphrase"]],
    )
    axes[0].set_title("Распределение классов MRPC")
    axes[0].set_ylabel("Количество")
    axes[1].hist(len1, bins=20, alpha=0.6, label="sentence1")
    axes[1].hist(len2, bins=20, alpha=0.6, label="sentence2")
    axes[1].set_title("Длины предложений (токены)")
    axes[1].set_xlabel("Токены")
    axes[1].set_ylabel("Частота")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> dict:
    """Запускает анализ MRPC и сохраняет результаты."""
    dataset = load_glue_mrpc_dataset("train")
    similarity, lengths, overlap, len1, len2 = analyze_dataset(dataset)
    plot_path = create_visualization((len1, len2, similarity["counts"]))
    result = {
        "similarity_distribution": similarity,
        "sentence_lengths": lengths,
        "word_overlap": overlap,
        "visualization_path": plot_path,
    }
    with open("mrpc_results.json", "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print("Готово: mrpc_results.json и visualization.png")
    return result


if __name__ == "__main__":
    main()
```

- GigaCode

```
К сожалению не сохранился чат
```
