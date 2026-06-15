using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Components;

namespace AILib.ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            string dataPath = @"C:\Users\user\Downloads\mnist_train.csv"; // Путь к вашему файлу

            Console.WriteLine("1. Чтение и подготовка данных...");
            var (inputs, targets, labels) = LoadData(dataPath);

            if (inputs.Count == 0)
            {
                Console.WriteLine("Данные не найдены!");
                return;
            }

            Console.WriteLine($"Загружено примеров: {inputs.Count}");

            // Инициализатор весов (небольшие случайные значения от -0.1 до 0.1)
            Random rnd = new Random(42);
            Func<float> weightInitializer = () => (float)(rnd.NextDouble() * 0.2 - 0.1);

            Console.WriteLine("2. Инициализация нейросети...");
            // Вход: 28x28 = 784 пикселя
            // Скрытый слой: 128 нейронов
            // Выход: 10 нейронов (вероятности для цифр 0-9)
            var net = new Net(
                inputDim: 784,
                hiddenLayers: new List<int> { 128 },
                outputDim: 10,
                init: weightInitializer,
                activation: new Relu()
            );

            int epochs = 150; // Количество проходов по всему датасету
            float learningRate = 0.01f;

            Console.WriteLine("3. Старт обучения...");

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                float totalLoss = 0f;
                int correctPredictions = 0;

                for (int i = 0; i < inputs.Count; i++)
                {
                    // Обучаем на одном примере
                    float loss = net.Train(inputs[i], targets[i], learningRate);
                    totalLoss += loss;

                    // Оценка (Predict) для подсчета точности
                    // В реальном коде обучение и валидацию лучше разделять на разные выборки
                    Span<float> prediction = new float[10];
                    net.Predict(inputs[i], prediction);

                    int predictedLabel = GetMaxIndex(prediction);
                    if (predictedLabel == labels[i])
                    {
                        correctPredictions++;
                    }
                }

                float averageLoss = totalLoss / inputs.Count;
                float accuracy = ((float)correctPredictions / inputs.Count) * 100f;
                float targetAccuracy = 90.0f;
                if (accuracy >= targetAccuracy)
                {
                    Console.WriteLine($"Достигнута требуемая точность ({accuracy:F2}%). Остановка обучения.");
                    break; // Выходим из цикла эпох
                }
                Console.WriteLine($"Эпоха {epoch:D2}/{epochs} | Loss: {averageLoss:F4} | Точность: {accuracy:F2}%");
            }

            Console.WriteLine("4. Сохранение модели...");
            string jsonModel = net.SaveToJson();
            File.WriteAllText("mnist_model.json", jsonModel);
            Console.WriteLine("Модель сохранена в mnist_model.json");
        }

        /// <summary>
        /// Парсит CSV файл, возвращает нормализованные входы, One-Hot таргеты и оригинальные лейблы.
        /// </summary>
        static (List<float[]> inputs, List<float[]> targets, List<int> labels) LoadData(string filePath)
        {
            var inputs = new List<float[]>();
            var targets = new List<float[]>();
            var labelsList = new List<int>();

            string[] lines = File.ReadAllLines(filePath);

            foreach (string line in lines)
            {
                if (string.IsNullOrWhiteSpace(line) || line.StartsWith("label"))
                    continue; // Пропускаем пустые строки и заголовок

                string[] parts = line.Split(',');

                int label = int.Parse(parts[0]);
                labelsList.Add(label);

                // One-Hot Encoding для выхода (10 классов)
                float[] target = new float[10];
                target[label] = 1f;
                targets.Add(target);

                // Нормализация пикселей (индексы с 1 по 784)
                float[] input = new float[784];
                for (int i = 0; i < 784; i++)
                {
                    // Переводим цвет [0..255] в диапазон [0.0..1.0]
                    input[i] = float.Parse(parts[i + 1]) / 255f;
                }
                inputs.Add(input);
            }

            return (inputs, targets, labelsList);
        }

        /// <summary>
        /// Возвращает индекс максимального элемента (предсказанный класс).
        /// </summary>
        static int GetMaxIndex(ReadOnlySpan<float> array)
        {
            int maxIndex = 0;
            float maxValue = array[0];
            for (int i = 1; i < array.Length; i++)
            {
                if (array[i] > maxValue)
                {
                    maxValue = array[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }
    }
}