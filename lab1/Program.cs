namespace lab1
{
    class Program
    {
        static void Main(string[] args)
        {
            #region Data Definition
            // Навчальна вибірка (15 прикладів)
            var trainData = new List<DataPoint>
            {
                new DataPoint(1, 7, 30, DroneType.Amateur), new DataPoint(2, 10, 50, DroneType.Amateur),
                new DataPoint(3, 12, 40, DroneType.Amateur), new DataPoint(4, 14, 60, DroneType.Amateur),
                new DataPoint(5, 15, 80, DroneType.Amateur), new DataPoint(6, 16, 70, DroneType.Amateur),
                new DataPoint(7, 18, 90, DroneType.Amateur), new DataPoint(8, 19, 95, DroneType.Amateur),
                new DataPoint(9, 22, 120, DroneType.Professional), new DataPoint(10, 25, 150, DroneType.Professional),
                new DataPoint(11, 28, 180, DroneType.Professional), new DataPoint(12, 30, 200, DroneType.Professional),
                new DataPoint(13, 32, 220, DroneType.Professional), new DataPoint(14, 35, 300, DroneType.Professional),
                new DataPoint(15, 27, 160, DroneType.Professional),
            };

            // Тестова вибірка (5 прикладів)
            var testData = new List<DataPoint>
            {
                new DataPoint(101, 21, 110, DroneType.Professional), new DataPoint(102, 19, 105, DroneType.Amateur),
                new DataPoint(103, 26, 90, DroneType.Amateur), new DataPoint(104, 15, 70, DroneType.Amateur),
                new DataPoint(105, 32, 250, DroneType.Professional),
            };

            // Зберігаємо копії оригінальних даних для візуалізації
            var originalTrainData = trainData.Select(d => new DataPoint(d.Id, d.Time, d.Altitude, d.Type)).ToList();
            var originalTestData = testData.Select(d => new DataPoint(d.Id, d.Time, d.Altitude, d.Type)).ToList();
            #endregion

            #region Normalization
            // Нормалізуємо обидві вибірки
            Normalizer.MinMaxNormalize(trainData, out double minTime, out double maxTime, out double minAlt, out double maxAlt);
            Normalizer.ApplyNormalization(testData, minTime, maxTime, minAlt, maxAlt);
            #endregion

            #region Menu and Training Setup
            // --- Меню вибору методу навчання ---
            Console.WriteLine("Choose a training method:");
            Console.WriteLine("1. Sequential (Base)");
            Console.WriteLine("2. Alternating Classes");
            Console.WriteLine("3. Random Shuffle");
            Console.Write("Enter your choice (1-3): ");
            string choice = Console.ReadLine();

            string trainingModeName;
            bool shuffleEachEpoch = false;
            int maxEpochs = 1000; // Стандартна кількість епох

            switch (choice)
            {
                case "1":
                    trainingModeName = "Sequential";
                    // Дані вже в послідовному порядку, нічого не робимо.
                    break;

                case "2":
                    trainingModeName = "Alternating Classes";
                    // Готуємо дані для поперемінної подачі
                    var amateurDrones = trainData.Where(d => d.Type == DroneType.Amateur).ToList();
                    var professionalDrones = trainData.Where(d => d.Type == DroneType.Professional).ToList();
                    var alternatingData = new List<DataPoint>();
                    int smallerCount = Math.Min(amateurDrones.Count, professionalDrones.Count);
                    for (int i = 0; i < smallerCount; i++)
                    {
                        alternatingData.Add(amateurDrones[i]);
                        alternatingData.Add(professionalDrones[i]);
                    }
                    alternatingData.AddRange(amateurDrones.Skip(smallerCount));
                    alternatingData.AddRange(professionalDrones.Skip(smallerCount));
                    trainData = alternatingData; // Перезаписуємо тренувальний набір
                    break;

                case "3":
                    trainingModeName = "Random Shuffle";
                    shuffleEachEpoch = true; // Встановлюємо прапорець для перемішування
                    maxEpochs = 10000; // Збільшуємо кількість епох в 10 разів
                    break;

                default:
                    Console.WriteLine("Invalid choice. Exiting.");
                    return;
            }
            #endregion

            #region Training Process
            // --- Процес навчання ---
            Console.WriteLine($"\n--- Starting Training: {trainingModeName} mode ---");

            // Використовуємо фіксований seed для відтворюваності результатів
            var perceptron = new Perceptron(learningRate: 0.01, randomSeed: null);
            var initialWeights = perceptron.GetWeights();
            Console.WriteLine($"Initial weights: w1={initialWeights.w1:F6}, w2={initialWeights.w2:F6}, b={initialWeights.b:F6}");

            int currentEpoch = 0;
            while (currentEpoch < maxEpochs)
            {
                // Якщо обрано випадковий режим, перемішуємо дані перед кожною епохою
                if (shuffleEachEpoch)
                {
                    trainData.Shuffle();
                }

                perceptron.TrainOneEpoch(trainData, out int errorsCount);
                Console.WriteLine($" Epoch {currentEpoch + 1}: errors = {errorsCount}");
                if (errorsCount == 0)
                {
                    Console.WriteLine("Training converged!");
                    break;
                }
                currentEpoch++;
            }
            #endregion

            #region Testing and Evaluation
            // --- Тестування та оцінка ---
            var (w1, w2, b) = perceptron.GetWeights();
            Console.WriteLine($"\nFinal weights: w1={w1:F6}, w2={w2:F6}, b={b:F6}");

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
            Console.WriteLine($"\nConfusion matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}");
            Console.WriteLine($"Accuracy = {accuracy:P2}");
            #endregion

            #region Visualization
            // --- Візуалізація ---
            List<DataPoint> allOriginalData = originalTrainData.Concat(originalTestData).ToList();
            DataVisualizer.PlotData(allOriginalData, (w1, w2, b), minTime, maxTime, minAlt, maxAlt);
            Console.WriteLine($"\nPlot saved to 'drone_classification_plot.png'");
            #endregion

            Console.WriteLine("\nDone. Press any key to exit...");
            Console.ReadKey();
        }
    }
}