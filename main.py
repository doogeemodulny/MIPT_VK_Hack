import Augmentation as Aug
import Dataset as Dt
import MachineLearning as ML
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, classification_report

# USE_EXTRA_YANDEX_DATASET = False # теперь не используется

train_df, val_df = Dt.CreateNecessaryDataset()
print('TrainDataFrame and ValidationDataFrame are loaded')
train_df, val_df, test_df = Dt.SplitIntoTest(train_df, val_df)  # Если нужен тестовый датасет
print('TrainDataFrame and ValidationDataFrame were split into TestDataFrame')

# Аугментирование
train_df = Aug.augment_relevant_answers_batch(train_df)

# Добавление английского (+ undersampled) датасета
train_df = Dt.AddMiraclEN(train_df)

Dt.SaveDF(train_df, val_df, test_df)  # Дабы сохранять датафреймы

trainingResults = ML.TrainingBertModel(train_df, val_df)
train_texts, train_labels, valid_texts, valid_labels = trainingResults[0]
model = trainingResults[1]
reference = 'best_model.pt'
ML.SaveModel(model, reference)
Model = ML.Model(reference)  # Подгрузка модели

# проверка на val_df
# predictions, classificationReport, rocAucScore = ML.Test1(valid_texts, valid_labels, Model)
# print("Results in validation:")
# print(classificationReport)
# print("roc auc score:", rocAucScore)

# проверка на test_df
testPredictions, testClassificationReport, testRocAucScore = ML.Test2(test_df, Model)
print("Results in test:")
print(testClassificationReport)
print("roc auc score:", testRocAucScore)
