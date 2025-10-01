using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

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
                new DataPoint(1,  7,  30,  DroneType.Amateur),
                new DataPoint(2, 10,  50,  DroneType.Amateur),
                new DataPoint(3, 12,  40,  DroneType.Amateur),
                new DataPoint(4, 14,  60,  DroneType.Amateur),
                new DataPoint(5, 15,  80,  DroneType.Amateur),
                new DataPoint(6, 16,  70,  DroneType.Amateur),
                new DataPoint(7, 18,  90,  DroneType.Amateur),
                new DataPoint(8, 19,  95,  DroneType.Amateur),
                new DataPoint(9, 22, 120,  DroneType.Professional),
                new DataPoint(10,25,150,  DroneType.Professional),
                new DataPoint(11,28,180,  DroneType.Professional),
                new DataPoint(12,30,200,  DroneType.Professional),
                new DataPoint(13,32,220,  DroneType.Professional),
                new DataPoint(14,35,300,  DroneType.Professional),
                new DataPoint(15,27,160,  DroneType.Professional),
            };

            // Тестова вибірка (5 прикладів)
            var testData = new List<DataPoint>
            {
                new DataPoint(101, 21, 110, DroneType.Professional),
                new DataPoint(102, 19, 105, DroneType.Amateur),
                new DataPoint(103, 26,  90,  DroneType.Amateur),
                new DataPoint(104, 15,  70,  DroneType.Amateur),
                new DataPoint(105, 32, 250, DroneType.Professional),
            };

            // Копії оригіналів (якщо потрібні пізніше для візуалізації/звіту)
            var originalTrainData = trainData.Select(d => new DataPoint(d.Id, d.Time, d.Altitude, d.Type)).ToList();
            var originalTestData = testData.Select(d => new DataPoint(d.Id, d.Time, d.Altitude, d.Type)).ToList();
            #endregion

            #region Normalization
            // Нормалізуємо тренувальні дані і застосовуємо ті ж min/max до тесту
            Normalizer.MinMaxNormalize(trainData, out double minTime, out double maxTime, out double minAlt, out double maxAlt);
            Normalizer.ApplyNormalization(testData, minTime, maxTime, minAlt, maxAlt);
            Console.WriteLine("Normalization done (min-max).");
            #endregion

            #region Training mode selection parameters
            // Параметри навчання
            int baseMaxEpochs = 100;   // базова кількість епох для sequential/alternating
            int randomSeed = 9;    // фіксований seed для відтворюваності (змінити на null-призначення на свій ризик)
            bool useSeed = false;       // увімкнути/вимкнути фіксований seed

            if(useSeed)
            {
                Console.Write("\nseed: ");
                string seedInput = Console.ReadLine();
                if (int.TryParse(seedInput, out int parsedSeed))
                {
                    randomSeed = parsedSeed;
                }
            }


            Console.WriteLine("\nChoose a training method:");
            Console.WriteLine("1. Sequential (Base)");
            Console.WriteLine("2. Alternating Classes");
            Console.WriteLine("3. Random sampling WITH replacement (N * 10 updates per virtual epoch)");
            Console.Write("Enter your choice (1-3): ");
            string choice = Console.ReadLine();
            #endregion

            #region Prepare working training set according to mode
            // робоча копія (щоб у випадку alternating не втікати оригінал)
            var workingTrain = trainData.Select(d => new DataPoint(d.Id, d.Time, d.Altitude, d.Type)).ToList();

            bool alternatingMode = false;
            bool samplingWithReplacement = false;
            string trainingModeName = "Sequential";

            switch (choice)
            {
                case "1":
                    trainingModeName = "Sequential";
                    break;

                case "2":
                    trainingModeName = "Alternating";
                    alternatingMode = true;
                    {
                        var amateur = workingTrain.Where(d => d.Type == DroneType.Amateur).ToList();
                        var professional = workingTrain.Where(d => d.Type == DroneType.Professional).ToList();
                        var alternating = new List<DataPoint>();
                        int smaller = Math.Min(amateur.Count, professional.Count);
                        for (int i = 0; i < smaller; i++)
                        {
                            alternating.Add(amateur[i]);
                            alternating.Add(professional[i]);
                        }
                        alternating.AddRange(amateur.Skip(smaller));
                        alternating.AddRange(professional.Skip(smaller));
                        workingTrain = alternating;
                    }
                    break;

                case "3":
                    trainingModeName = "RandomSamplingWithReplacement";
                    samplingWithReplacement = true;
                    break;

                default:
                    Console.WriteLine("Invalid choice. Exiting.");
                    return;
            }

            Console.WriteLine($"\nSelected training mode: {trainingModeName}");
            #endregion

            #region Initialize perceptron
            var perceptron = useSeed ? new Perceptron(learningRate: 0.01, randomSeed: randomSeed)
                                     : new Perceptron(learningRate: 0.01, randomSeed: null);

            var (initW1, initW2, initB) = perceptron.GetWeights();
            Console.WriteLine($"\nInitial weights: w1={initW1:F6}, w2={initW2:F6}, b={initB:F6}");
            #endregion

            #region Training loops
            Console.WriteLine("\n--- Training started ---");

            // Що з означенням "збільшити кількість ітерацій в 10 раз" — реалізуємо як
            // sampling-with-replacement: за одну "virtual epoch" робимо N * 10 оновлень (N = розмір train)
            int N = workingTrain.Count;
            int virtualEpochs = baseMaxEpochs; // кількість virtual epochs (можна змінити)
            var rng = useSeed ? new Random(randomSeed) : new Random();

            // Зберігаємо історію помилок (за епохами) тимчасово як список (не зберігаємо в файл зараз)
            var epochErrors = new List<int>();

            if (samplingWithReplacement)
            {
                // sampling-with-replacement mode
                for (int ve = 0; ve < virtualEpochs; ve++)
                {
                    int errorsThisVirtualEpoch = 0;
                    int updatesPerVirtualEpoch = N * 10; // 10x ітерацій
                    for (int k = 0; k < updatesPerVirtualEpoch; k++)
                    {
                        int idx = rng.Next(N); // індекс у workingTrain
                        var sample = workingTrain[idx];
                        bool updated = perceptron.UpdateOneSample(sample);
                        if (updated) errorsThisVirtualEpoch++;
                    }

                    epochErrors.Add(errorsThisVirtualEpoch);
                    Console.WriteLine($" Virtual epoch {ve + 1}/{virtualEpochs}: updates = {N * 10}, errors = {errorsThisVirtualEpoch}");
                    if (errorsThisVirtualEpoch == 0)
                    {
                        Console.WriteLine("Converged during sampling-with-replacement.");
                        break;
                    }
                }
            }
            else
            {
                // sequential or alternating: класичні епохи (по одному проходу по workingTrain)
                for (int e = 0; e < baseMaxEpochs; e++)
                {
                    perceptron.TrainOneEpoch(workingTrain, out int errorsCount);
                    epochErrors.Add(errorsCount);
                    Console.WriteLine($" Epoch {e + 1}/{baseMaxEpochs}: errors = {errorsCount}");
                    if (errorsCount == 0)
                    {
                        Console.WriteLine("Converged.");
                        break;
                    }
                }
            }

            Console.WriteLine("--- Training finished ---");
            #endregion

            #region Final weights and evaluation
            var (w1, w2, b) = perceptron.GetWeights();
            Console.WriteLine($"\nFinal weights: w1={w1:F6}, w2={w2:F6}, b={b:F6}");

            // Оцінка на train (workingTrain) та testData
            var trainMetrics = EvaluateMetrics(workingTrain, perceptron);
            var testMetrics = EvaluateMetrics(testData, perceptron);

            Console.WriteLine("\n--- Metrics ---");
            Console.WriteLine("On TRAIN set:");
            PrintMetrics(trainMetrics);
            Console.WriteLine("\nOn TEST set:");
            PrintMetrics(testMetrics);
            #endregion

            Console.WriteLine("\nDone. Press any key to exit...");
            Console.ReadKey();
        }

        // Повертає TP,TN,FP,FN, Accuracy, Precision, Recall, F1
        static (int TP, int TN, int FP, int FN, double Accuracy, double Precision, double Recall, double F1) EvaluateMetrics(List<DataPoint> data, Perceptron p)
        {
            int tp = 0, tn = 0, fp = 0, fn = 0;
            foreach (var d in data)
            {
                int y = p.Predict(d);
                if (d.Label == 1 && y == 1) tp++;
                else if (d.Label == 0 && y == 0) tn++;
                else if (d.Label == 0 && y == 1) fp++;
                else if (d.Label == 1 && y == 0) fn++;
            }

            int total = tp + tn + fp + fn;
            double acc = total == 0 ? 0 : (tp + tn) / (double)total;
            double prec = (tp + fp) == 0 ? 0 : tp / (double)(tp + fp);
            double rec = (tp + fn) == 0 ? 0 : tp / (double)(tp + fn);
            double f1 = (prec + rec) == 0 ? 0 : 2 * prec * rec / (prec + rec);
            return (tp, tn, fp, fn, acc, prec, rec, f1);
        }

        static void PrintMetrics((int TP, int TN, int FP, int FN, double Accuracy, double Precision, double Recall, double F1) m)
        {
            Console.WriteLine($" TP={m.TP}, TN={m.TN}, FP={m.FP}, FN={m.FN}");
            Console.WriteLine($" Accuracy = {m.Accuracy:P2}");
            Console.WriteLine($" Precision = {m.Precision:P2}");
            Console.WriteLine($" Recall = {m.Recall:P2}");
            Console.WriteLine($" F1-score = {m.F1:P2}");
        }
    }
}
