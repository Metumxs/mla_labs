using System;
using System.Collections.Generic;
using System.Linq;

namespace lab1
{
    class Program
    {
        static void Main(string[] args)
        {
            // Навчальна вибірка (15 прикладів)
            var trainData = new List<DataPoint>
      {
        new DataPoint(1, 7, 30, DroneType.Amateur),
        new DataPoint(2, 10, 50, DroneType.Amateur),
        new DataPoint(3, 12, 40, DroneType.Amateur),
        new DataPoint(4, 14, 60, DroneType.Amateur),
        new DataPoint(5, 15, 80, DroneType.Amateur),
        new DataPoint(6, 16, 70, DroneType.Amateur),
        new DataPoint(7, 18, 90, DroneType.Amateur),
        new DataPoint(8, 19, 95, DroneType.Amateur),
        new DataPoint(9, 22, 120, DroneType.Professional),
        new DataPoint(10,25, 150, DroneType.Professional),
        new DataPoint(11,28, 180, DroneType.Professional),
        new DataPoint(12,30, 200, DroneType.Professional),
        new DataPoint(13,32, 220, DroneType.Professional),
        new DataPoint(14,35, 300, DroneType.Professional),
        new DataPoint(15,27, 160, DroneType.Professional),
      };

            // Тестова вибірка (5 прикладів)
            var testData = new List<DataPoint>
      {
        new DataPoint(101, 21, 110, DroneType.Professional),
        new DataPoint(102, 19, 105, DroneType.Amateur),
        new DataPoint(103, 26, 90, DroneType.Amateur),
        new DataPoint(104, 15, 70, DroneType.Amateur),
        new DataPoint(105, 32, 250, DroneType.Professional),
      };

            // Копії оригінальних даних
            var originalTrainData = trainData.Select(d => new DataPoint(d.Id, d.Time, d.Altitude, d.Type)).ToList();
            var originalTestData = testData.Select(d => new DataPoint(d.Id, d.Time, d.Altitude, d.Type)).ToList();

            //
            // Вивід початкових даних
            //
            Console.WriteLine("Original Data:");
            Console.WriteLine("--- Training Set ---");
            originalTrainData.ForEach(d => Console.WriteLine(d));
            Console.WriteLine("--- Test Set ---");
            originalTestData.ForEach(d => Console.WriteLine(d));

            //
            // Нормалізація
            //
            Normalizer.MinMaxNormalize(trainData, out double minTime, out double maxTime, out double minAlt, out double maxAlt);
            Normalizer.ApplyNormalization(testData, minTime, maxTime, minAlt, maxAlt);

            //
            // Вивід параметрів нормалізації та нормалізованих даних
            //
            Console.WriteLine("\nNormalization Parameters:");
            Console.WriteLine($" minTime={minTime}, maxTime={maxTime}");
            Console.WriteLine($" minAlt={minAlt}, maxAlt={maxAlt}");
            Console.WriteLine("\nNormalized Data:");
            Console.WriteLine("--- Training Set ---");
            trainData.ForEach(d => Console.WriteLine(d));
            Console.WriteLine("--- Test Set ---");
            testData.ForEach(d => Console.WriteLine(d));

            // Генерація випадкового цілого числа.
            // Це буде наш сід для відтворюваності.
            //var seedGenerator = new Random();
            //int mySeed = seedGenerator.Next();
            //Console.WriteLine($"\nGenerated Random Seed: {mySeed}");

            //
            // Ініціалізація перцептрона та вивід початкових ваг
            //
            var perceptron = new Perceptron(learningRate: 0.01, randomSeed: 790800079); // Використання seed для відтворюваності
            var initialWeights = perceptron.GetWeights();
            Console.WriteLine($"\nInitial weights: w1={initialWeights.w1:F6}, w2={initialWeights.w2:F6}, b={initialWeights.b:F6}");

            //
            // Навчання
            //
            int currentEpoch = 0;
            const int maxEpochs = 1000;
            Console.WriteLine("\nTraining:");
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
            //
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
            List<DataPoint> allOriginalData = originalTrainData.Concat(originalTestData).ToList();
            DataVisualizer.PlotData(allOriginalData, (w1, w2, b), minTime, maxTime, minAlt, maxAlt);
            Console.WriteLine($"\nPlot saved to 'drone_classification_plot.png'");

            // Кінець
            Console.WriteLine("\nDone. Press any key to exit...");
            Console.ReadKey();
        }
    }
}