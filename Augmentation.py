'''Пока в разработке'''
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
import pymorphy3
import random
import torch

def apply_augmentations(df,
                        symmetrize=True,
                        speech_garbage=0,
                        drop_symbol=0,
                        drop_token=0,
                        double_token=0,
                        insert_random_symbol=0,
                        swap_tokens=0,
                        siblings=0):

    cyrillic_letters = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
    sibling_letters = { 'а': 'a','В': 'B', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'Т': 'T', 'у': 'y', 'х': 'x'}


    augmentation_probas = [speech_garbage, drop_symbol, drop_token, double_token, insert_random_symbol, swap_tokens, siblings]

    if sum(augmentation_probas) <= 0:
        if symmetrize:
            raise RuntimeError("To symmetrize the classes at least one augmentation must be applied")
        else:
            return df


    df_0 = df[df["isRelevant"] == 0]
    df_1 = df[df["isRelevant"] == 1]
    new_rows = []
    if symmetrize:
        while df_0.shape[0] + len(new_rows) < df_1.shape[0]:
            df_0 = df_0.drop_duplicates()
            if speech_garbage > 0:
                if np.random.rand() < speech_garbage:
                    new_row = df_0.sample().copy()
                    text = new_row['question']
                    text_split = text.iloc[0].split()
                    if text_split:
                        insert_index = np.random.randint(0, len(text_split) + 1)
                        random_word = np.random.choice(["ээ", "мм", "ну", "кхм-кхм"])
                        text_split.insert(insert_index, random_word)
                    text = " ".join(text_split)
                    new_row["question"] = text
                    if text_split:
                      if type(new_row["question"].iloc[0]) != type("aboba"):
                          print(type(new_row["question"].iloc[0]), 1)
                      new_rows.append(new_row)

            if drop_symbol > 0:
                if np.random.rand() < drop_symbol:
                    new_row = df_0.sample().copy()
                    for _ in range (10):
                        symbol_to_drop = np.random.choice(list(cyrillic_letters))
                        new_row['answer'] = new_row['answer'].str.replace(symbol_to_drop, '', regex=False)
                    if type(new_row["answer"].iloc[0]) != type("aboba"):
                        print(type(new_row["answer"].iloc[0]), 2)
                    new_rows.append(new_row)

            if drop_token > 0:
                if np.random.rand() < drop_token:
                    new_row = df_0.sample().copy()
                    tokens = new_row['answer'].iloc[0].split()
                    drop_index = np.random.randint(0, len(tokens) + 1)
                    if len(tokens) > 0:
                        drop_index = np.random.randint(0, len(tokens))
                        tokens.pop(drop_index)
                    new_row['answer'] = ' '.join(tokens)
                    if tokens:
                      if type(new_row["answer"].iloc[0]) != type("aboba"):
                          print(type(new_row["answer"].iloc[0]), 3)
                      new_rows.append(new_row)

            if double_token > 0:
                if np.random.rand() < double_token:
                    new_row = df_0.sample().copy()
                    tokens = new_row['answer'].iloc[0].split()
                    for _ in range(5):
                        if tokens:
                            duplicate_index = np.random.randint(0, len(tokens))
                            tokens.insert(duplicate_index + 1, tokens[duplicate_index])
                    new_row['answer'] = ' '.join(tokens)
                    if tokens:
                        if type(new_row["answer"].iloc[0]) != type("aboba"):
                            print(type(new_row["answer"].iloc[0]), 4)
                        new_rows.append(new_row)

            if insert_random_symbol > 0:
                if np.random.rand() < insert_random_symbol:
                    new_row = df_0.sample().copy()
                    for _ in range(10):
                        random_symbol = np.random.choice(list(cyrillic_letters))
                        insert_index = np.random.randint(0, len(new_row['answer']))
                        new_row['answer'] = new_row['answer'][:insert_index] + random_symbol + new_row['answer'][insert_index:]

                    if type(new_row["answer"].iloc[0]) != type("aboba"):
                        print(type(new_row["answer"].iloc[0]), 5)
                    new_rows.append(new_row)

            if swap_tokens > 0:
                if np.random.rand() < swap_tokens:
                    new_row = df_0.sample().copy()
                    tokens = new_row['answer'].iloc[0].split()
                    for _ in range(3):
                        if len(tokens) > 1:
                            swap_index = np.random.randint(0, len(tokens) - 1)
                            tokens[swap_index], tokens[swap_index + 1] = tokens[swap_index + 1], tokens[swap_index]
                    new_row['answer'] = ' '.join(tokens)
                    if tokens:
                        if type(new_row["answer"].iloc[0]) != type("aboba"):
                            print(type(new_row["answer"].iloc[0]), 6)
                        new_rows.append(new_row)


            if siblings > 0:
                if np.random.rand() < siblings:
                    new_row = df_0.sample().copy()
                    answer = new_row['answer'].iloc[0]
                    new_answer = []

                    for char in answer:
                        if char in sibling_letters and np.random.rand() < siblings:
                            new_answer.append(sibling_letters[char])
                        else:
                            new_answer.append(char)

                    new_row['answer'] = ''.join(new_answer)
                    if new_answer:
                        if type(new_row["answer"].iloc[0]) != type("aboba"):
                            print(type(new_row["answer"].iloc[0]), 7)
                        new_rows.append(new_row)


        while df_1.shape[0] + len(new_rows) < df_0.shape[0]:
            if speech_garbage > 0:
                if np.random.rand() < speech_garbage:
                    new_row = df_1.sample().copy()
                    text = new_row['question']
                    text_split = text.iloc[0].split()
                    if text_split:
                        insert_index = np.random.randint(0, len(text_split) + 1)
                        random_word = np.random.choice(["ээ", "мм", "ну", "кхм-кхм"])
                        text_split.insert(insert_index, random_word)
                    text = " ".join(text_split)
                    new_row["question"] = text
                    if text_split:
                        if type(new_row["question"].iloc[0]) != type("aboba"):
                            print(type(new_row["question"].iloc[0]), 8)
                        new_rows.append(new_row)

            if drop_symbol > 0:
                if np.random.rand() < drop_symbol:
                    new_row = df_1.sample().copy()
                    for _ in range (10):
                        symbol_to_drop = np.random.choice(list(cyrillic_letters))
                        new_row['answer'] = new_row['answer'].str.replace(symbol_to_drop, '', regex=False)
                    if type(new_row["answer"].iloc[0]) != type("aboba"):
                        print(type(new_row["answer"].iloc[0]), 9)
                    new_rows.append(new_row)

            if drop_token > 0:
                if np.random.rand() < drop_token:
                    new_row = df_1.sample().copy()
                    tokens = new_row['answer'].iloc[0].split()
                    drop_index = np.random.randint(0, len(tokens) + 1)
                    if len(tokens) > 0:
                        drop_index = np.random.randint(0, len(tokens))
                        tokens.pop(drop_index)
                    new_row['answer'] = ' '.join(tokens)
                    if tokens:
                        if type(new_row["answer"].iloc[0]) != type("aboba"):
                            print(type(new_row["answer"].iloc[0]), 10)
                        new_rows.append(new_row)

            if double_token > 0:
                if np.random.rand() < double_token:
                    new_row = df_1.sample().copy()
                    tokens = new_row['answer'].iloc[0].split()
                    for _ in range(5):
                        if tokens:
                            duplicate_index = np.random.randint(0, len(tokens))
                            tokens.insert(duplicate_index + 1, tokens[duplicate_index])
                    new_row['answer'] = ' '.join(tokens)
                    if tokens:
                        if type(new_row["answer"].iloc[0]) != type("aboba"):
                            print(type(new_row["answer"].iloc[0]), 11)
                        new_rows.append(new_row)

            if insert_random_symbol > 0:
                if np.random.rand() < insert_random_symbol:
                    new_row = df_1.sample().copy()
                    for _ in range(10):
                        random_symbol = np.random.choice(list(cyrillic_letters))
                        insert_index = np.random.randint(0, len(new_row['answer']))
                        new_row['answer'] = new_row['answer'].str[:insert_index] + random_symbol + new_row['answer'].str[insert_index:]
                    if type(new_row["answer"].iloc[0]) != type("aboba"):
                        print(type(new_row["answer"].iloc[0]), 12)
                    new_rows.append(new_row)

            if swap_tokens > 0:
                if np.random.rand() < swap_tokens:
                    new_row = df_1.sample().copy()
                    tokens = new_row['answer'].iloc[0].split()
                    for _ in range(3):
                        if len(tokens) > 1:
                            swap_index = np.random.randint(0, len(tokens) - 1)
                            tokens[swap_index], tokens[swap_index + 1] = tokens[swap_index + 1], tokens[swap_index]
                    new_row['answer'] = ' '.join(tokens)
                    if tokens:
                        if type(new_row["answer"].iloc[0]) != type("aboba"):
                            print(type(new_row["answer"].iloc[0]), 13)
                        new_rows.append(new_row)

            if siblings > 0:
                if np.random.rand() < siblings:
                    new_row = df_1.sample().copy()
                    answer = new_row['answer'].iloc[0]
                    new_answer = []

                    for char in answer:
                        if char in sibling_letters and np.random.rand() < siblings:
                            new_answer.append(sibling_letters[char])
                        else:
                            new_answer.append(char)

                    new_row['answer'] = ''.join(new_answer)
                    if new_answer:
                        if type(new_row["answer"].iloc[0]) != type("aboba"):
                            print(type(new_row["answer"].iloc[0]), 14)
                        new_rows.append(new_row)

    if new_rows:
        new_df = pd.concat(new_rows).reset_index(drop=True)
        df = pd.concat([df, new_df], ignore_index=True)
    df = df.drop_duplicates()

    return df

class Augmentator:

    def __init__(self):
        try:
            self. russian_stopwords = stopwords.words("russian")
        except LookupError:
            nltk.download('stopwords')
            self.russian_stopwords = stopwords.words("russian")

        try:
            word_tokenize("Пример текста")
        except LookupError:
            nltk.download('punkt')

        self.morph = pymorphy3.MorphAnalyzer()

        # Проверка доступности CUDA
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Используется CUDA")
        else:
            device = torch.device("cpu")
            print("CUDA не доступна, используется CPU")

        # device = torch.device("cpu")

        # Инициализируем пайплайн для перевода
        self.translator = pipeline("translation_ru_to_en", model="Helsinki-NLP/opus-mt-ru-en", device=device)
        self.back_translator = pipeline("translation_en_to_ru", model="Helsinki-NLP/opus-mt-en-ru", device=device)

    def augment_answers(self, answer, num_augmentations=1):
        augmented_answers = []
        for _ in range(num_augmentations):
            augmented_answer = answer
            # Back Translation
            translated = self.translator(augmented_answer, batch_size=16)[0]["translation_text"]
            back_translated = self.back_translator(translated, batch_size=16)[0]["translation_text"]
            augmented_answers.append(back_translated)

        return augmented_answers

    def augment_questions(self, question, num_augmentations=2):
        augmented_questions = []
        for _ in range(num_augmentations):
            augmented_question = question
            # Back Translation (с большей вероятностью, чем для ответов)
            translated = self.translator(augmented_question, batch_size=16)[0]["translation_text"]
            back_translated = self.back_translator(translated, batch_size=16)[0]["translation_text"]
            augmented_questions.append(back_translated)

            # Synonym Replacement (требует словарь синонимов)
            words = word_tokenize(augmented_question)
            for i in range(len(words)):
                if words[i] not in self.russian_stopwords:
                    try:
                        synonyms = self.morph.parse(words[i])[0].lexeme
                        if synonyms:
                            synonym = random.choice([s.word for s in synonyms])
                            words[i] = synonym
                    except:
                        pass
            augmented_questions.append(" ".join(words))

            # Random Insertion/Deletion
            words = word_tokenize(augmented_question)
            if len(words) > 1:
                # if random.random() < 0.5:  # Insertion
                if random.random() < 0:  # пока так, без Insertion
                    insert_index = random.randint(0, len(words))
                    # Вставка случайного слова (нужен словарь слов)
                    # words.insert(insert_index, random.choice(russian_words))
                else:  # Deletion
                    del words[random.randint(0, len(words) - 1)]
            augmented_questions.append(" ".join(words))

        return augmented_questions

'''# Пример использования:
A=Augmentator()
example_question = "Кто первым полетел в космос?"
augmented_questions = A.augment_questions(example_question)
print(f"Аугментированные вопросы: {augmented_questions}")

example_answer = "Первым космонавтом стал Юрий Алексеевич Гагарин из СССР. Он полетел в космос 12 апреля 1961 года."
augmented_answers = A.augment_answers(example_answer)
print(f"Аугментированные ответы: {augmented_answers}")'''

def apply_augmentations_df(df):
    new_rows = []
    processed_questions = {}  # Словарь для хранения обработанных вопросов
    A=Augmentator()

    for i in range(len(df)):
        question = df.loc[i, "question"]
        answer = df.loc[i, "answer"]

        if i % 10 == 0: print(i)

        # Проверяем длину ответа
        if len(answer) > 550:
            continue  # Пропускаем ответ, если он слишком длинный

        # Проверяем, был ли вопрос уже обработан
        if question not in processed_questions:
            # Аугментация вопроса
            augmented_questions = A.augment_questions(question)
            processed_questions[question] = augmented_questions
        else:
            augmented_questions = processed_questions[question]

        # Аугментация ответа
        augmented_answers = A.augment_answers(df.loc[i, "answer"])

        for q in augmented_questions:
            for a in augmented_answers:
                new_rows.append({'question': q, 'answer': a, 'isRelevant': df.loc[i, "isRelevant"]})

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df

# train_df = apply_augmentations_df(train_df)
# слишком долго работает, надо оптимизировать