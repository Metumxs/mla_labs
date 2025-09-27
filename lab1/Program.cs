using System;
using System.Collections.Generic;

namespace lab1
{
    class Program
    {
        static void Main(string[] args)
        {
            // Навчальна вибірка (15 прикладів) — Id у межах 1..15
            var trainData = new List<DataPoint>
            {
                new DataPoint(1,  7,  30,  DroneType.Amateur),
                new DataPoint(2, 10,  50,  DroneType.Amateur),
                new DataPoint(3, 12,  40,  DroneType.Amateur),
                new DataPoint(4, 14,  60,  DroneType.Amateur),
                new DataPoint(5, 15,  80,  DroneType.Amateur),
                new DataPoint(6, 16,  70,  DroneType.Amateur),
                new DataPoint(7, 18,  90,  DroneType.Amateur),
                new DataPoint(8, 19,  95,  DroneType.Amateur),
                new DataPoint(9, 22, 120,  DroneType.Professional),
                new DataPoint(10,25, 150,  DroneType.Professional),
                new DataPoint(11,28, 180,  DroneType.Professional),
                new DataPoint(12,30, 200,  DroneType.Professional),
                new DataPoint(13,32, 220,  DroneType.Professional),
                new DataPoint(14,35, 300,  DroneType.Professional),
                new DataPoint(15,27, 160,  DroneType.Professional),
            };

            // Тестова вибірка (5 прикладів) — Id поза діапазоном навчальних у межах 101..105
            var testData = new List<DataPoint>
            {
                new DataPoint(101, 21, 110, DroneType.Professional),
                new DataPoint(102, 19, 105, DroneType.Amateur),
                new DataPoint(103, 26,  90,  DroneType.Amateur),
                new DataPoint(104, 15,  70,  DroneType.Amateur),
                new DataPoint(105, 32, 250, DroneType.Professional),
            };

            // Копії оригінальних даних для подальшої візуалізації (до нормалізації)
            var originalTrainData = trainData.Select(d => new DataPoint(d.Id, d.Time, d.Altitude, d.Type)).ToList();
            var originalTestData = testData.Select(d => new DataPoint(d.Id, d.Time, d.Altitude, d.Type)).ToList();

            // Нормалізація (min-max) — на основі навчальної вибірки
            Normalizer.MinMaxNormalize(trainData, out double minTime, out double maxTime, out double minAlt, out double maxAlt);

            // Нормалізувати тестову вибірку за тими ж min/max
            Normalizer.ApplyNormalization(testData, minTime, maxTime, minAlt, maxAlt);

            // Ініціалізація перцептрона (встановлено learningRate; можна передавати seed для репродукції)
            var perceptron = new Perceptron(learningRate: 0.01, randomSeed: null);
            
            //
            // Навчання
            int currentEpoch = 0;
            const int maxEpochs = 1000;
            Console.WriteLine("Training:");
            while (currentEpoch < maxEpochs)
            {
                perceptron.TrainOneEpoch(trainData, out int errorsСount);
                Console.WriteLine($" Epoch {currentEpoch + 1}: errors = {errorsСount}");
                if (errorsСount == 0) break;
                ++currentEpoch;
            }

            // Фінальні ваги
            var (w1, w2, b) = perceptron.GetWeights();
            Console.WriteLine($"\nFinal weights: w1={w1:F6}, w2={w2:F6}, b={b:F6}");

            //
            // Тестування
            Console.WriteLine("\nTest results:");
            int tp = 0, tn = 0, fp = 0, fn = 0;

            foreach (var d in testData)
            {
                int y = perceptron.Predict(d);
                var predictedType = y == 1 ? DroneType.Professional : DroneType.Amateur;
                Console.WriteLine($"[{d.Id}] true={d.Type}, pred={predictedType}");

                if (d.Type == DroneType.Professional && predictedType == DroneType.Professional) ++tp;
                if (d.Type == DroneType.Amateur && predictedType == DroneType.Amateur) ++tn;
                if (d.Type == DroneType.Amateur && predictedType == DroneType.Professional) ++fp;
                if (d.Type == DroneType.Professional && predictedType == DroneType.Amateur) ++fn;
            }

            double accuracy = (tp + tn) / (double)(tp + tn + fp + fn);
            Console.WriteLine($"\nConfusion matrix:");
            Console.WriteLine($"TP={tp}, TN={tn}, FP={fp}, FN={fn}");
            Console.WriteLine($"Accuracy = {accuracy:P2}");

            //
            // Візуалізація
            //
            // Об'єднуємо всі вихідні (денормалізовані) дані в один список
            List<DataPoint> allOriginalData = originalTrainData.Concat(originalTestData).ToList();

            // Викликаємо метод для створення графіка
            DataVisualizer.PlotData(allOriginalData, (w1, w2, b), minTime, maxTime, minAlt, maxAlt);
            Console.WriteLine($"\nPlot saved to 'drone_classification_plot.png'");

            // Кінець
            Console.WriteLine("\nDone. Press any key to exit...");
            Console.ReadKey();
        }
    }
}
