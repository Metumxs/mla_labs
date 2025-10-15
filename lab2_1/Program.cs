using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using ScottPlot;

namespace lab2_1
{
    // Клас для зберігання вхідних даних
    public class RegressionInput
    {
        [LoadColumn(0)]
        public float X { get; set; }

        [LoadColumn(1)]
        public float Y { get; set; }
    }

    // Клас для передбачень
    public class RegressionPrediction
    {
        [ColumnName("Score")]
        public float PredictedY { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.WriteLine("=== Лабораторна робота 2: Завдання 1 ===");
            Console.WriteLine("Варіант 2: y = (x - 3)² + (x + 1)/(x² - 1)\n");

            // Крок 1: Генерація синтетичних даних
            Console.WriteLine("[1] Генерація 100 точок даних з шумом...");
            var allData = GenerateSyntheticData(100, -3, 5, noiseLevel: 0.5);

            // Крок 2: Візуалізація початкових даних
            Console.WriteLine("[2] Візуалізація початкових даних...");
            VisualizeInitialData(allData, -3, 5);

            // Крок 3: Розділення на train/test (80/20)
            Console.WriteLine("[3] Розділення на навчальну (80%) та тестову (20%) вибірки...");
            var (trainData, testData) = SplitData(allData, trainRatio: 0.8);
            Console.WriteLine($"   Навчальна вибірка: {trainData.Count} точок");
            Console.WriteLine($"   Тестова вибірка: {testData.Count} точок\n");

            // Крок 4-7: Створення, навчання та тестування моделі
            var mlContext = new MLContext(seed: 42);

            Console.WriteLine("[4] Створення моделі регресії...");
            Console.WriteLine("   Архітектура:");
            Console.WriteLine("   - Алгоритм: LightGBM (Light Gradient Boosting Machine)");
            Console.WriteLine("   - Кількість дерев: 100");
            Console.WriteLine("   - Кількість листків: 31");
            Console.WriteLine("   - Learning rate: 0.1");
            Console.WriteLine("   - Функція втрат: Squared Loss (MSE)\n");

            // Завантаження даних
            var trainDataView = mlContext.Data.LoadFromEnumerable(trainData);
            var testDataView = mlContext.Data.LoadFromEnumerable(testData);

            // Створення pipeline з LightGBM
            var pipeline = mlContext.Transforms.Concatenate("Features", nameof(RegressionInput.X))
                .Append(mlContext.Regression.Trainers.LightGbm(
                    labelColumnName: nameof(RegressionInput.Y),
                    featureColumnName: "Features",
                    numberOfLeaves: 31,
                    numberOfIterations: 100,
                    learningRate: 0.1,
                    minimumExampleCountPerLeaf: 5));

            Console.WriteLine("[5] Навчання моделі...");
            var model = pipeline.Fit(trainDataView);
            Console.WriteLine("   ✓ Навчання завершено\n");

            // Крок 8: Передбачення та оцінка
            Console.WriteLine("[6] Оцінка якості моделі...");

            // Передбачення на навчальних даних
            var trainPredictions = model.Transform(trainDataView);
            var trainMetrics = mlContext.Regression.Evaluate(trainPredictions,
                labelColumnName: nameof(RegressionInput.Y));

            // Передбачення на тестових даних
            var testPredictions = model.Transform(testDataView);
            var testMetrics = mlContext.Regression.Evaluate(testPredictions,
                labelColumnName: nameof(RegressionInput.Y));

            // Крок 9: Розрахунок та виведення R²
            Console.WriteLine("\n=== РЕЗУЛЬТАТИ ===");
            Console.WriteLine($"R² (навчальна вибірка): {trainMetrics.RSquared:F4}");
            Console.WriteLine($"R² (тестова вибірка):   {testMetrics.RSquared:F4}");
            Console.WriteLine($"MAE (тестова):          {testMetrics.MeanAbsoluteError:F4}");
            Console.WriteLine($"RMSE (тестова):         {testMetrics.RootMeanSquaredError:F4}\n");

            // Інтерпретація результатів
            InterpretR2(trainMetrics.RSquared, testMetrics.RSquared);

            // Крок 10: Візуалізація результатів
            Console.WriteLine("\n[7] Візуалізація результатів передбачення...");
            VisualizeResults(model, mlContext, testData, -3, 5);

            Console.WriteLine("\n✓ Завдання виконано! Графіки збережено у папці проекту.");
            Console.WriteLine("  - initial_data.png - початкові дані");
            Console.WriteLine("  - predictions.png - результати передбачення");
        }

        // Обчислення цільової функції
        static double TargetFunction(double x)
        {
            // y = (x - 3)² + (x + 1)/(x² - 1)
            // Уникаємо точок x = ±1 (розриви)
            if (Math.Abs(Math.Abs(x) - 1) < 0.1)
                return double.NaN;

            return Math.Pow(x - 3, 2) + (x + 1) / (x * x - 1);
        }

        // Функція генерації синтетичних даних
        static List<RegressionInput> GenerateSyntheticData(int count, double xMin, double xMax, double noiseLevel)
        {
            var random = new Random(42);
            var data = new List<RegressionInput>();
            int attempts = 0;
            int maxAttempts = count * 10;

            while (data.Count < count && attempts < maxAttempts)
            {
                attempts++;

                // Рівномірний розподіл x
                double x = xMin + (xMax - xMin) * random.NextDouble();

                // Пропускаємо точки поблизу розривів x = ±1
                if (Math.Abs(Math.Abs(x) - 1) < 0.15)
                    continue;

                // Обчислення ідеального y за формулою варіанту 2
                double yIdeal = TargetFunction(x);

                // Перевірка на екстремальні значення
                if (double.IsNaN(yIdeal) || double.IsInfinity(yIdeal) || Math.Abs(yIdeal) > 50)
                    continue;

                // Додавання нормального шуму
                double noise = NormalRandom(random) * noiseLevel;
                double yNoisy = yIdeal + noise;

                data.Add(new RegressionInput { X = (float)x, Y = (float)yNoisy });
            }

            return data;
        }

        // Генерація нормально розподіленої випадкової величини (Box-Muller transform)
        static double NormalRandom(Random random)
        {
            double u1 = random.NextDouble();
            double u2 = random.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        // Розділення даних на навчальну та тестову вибірки
        static (List<RegressionInput> train, List<RegressionInput> test) SplitData(
            List<RegressionInput> data, double trainRatio)
        {
            var random = new Random(42);
            var shuffled = data.OrderBy(x => random.Next()).ToList();

            int trainCount = (int)(shuffled.Count * trainRatio);
            var train = shuffled.Take(trainCount).ToList();
            var test = shuffled.Skip(trainCount).ToList();

            return (train, test);
        }

        // Візуалізація початкових даних
        static void VisualizeInitialData(List<RegressionInput> data, double xMin, double xMax)
        {
            var plt = new ScottPlot.Plot();

            // Точки даних (з шумом)
            double[] dataX = data.Select(d => (double)d.X).ToArray();
            double[] dataY = data.Select(d => (double)d.Y).ToArray();
            var scatter = plt.Add.Scatter(dataX, dataY);
            scatter.Color = ScottPlot.Color.FromHex("#0000FF");
            scatter.MarkerSize = 8;
            scatter.LineWidth = 0;
            scatter.LegendText = "Дані з шумом";

            // Ідеальна функція y = (x - 3)² + (x + 1)/(x² - 1)
            // Малюємо окремо частини (уникаючи розривів)

            // Ліва частина: від xMin до -1.15
            var leftX = new List<double>();
            var leftY = new List<double>();
            for (double x = xMin; x < -1.15; x += 0.02)
            {
                double y = TargetFunction(x);
                if (!double.IsNaN(y) && Math.Abs(y) < 50)
                {
                    leftX.Add(x);
                    leftY.Add(y);
                }
            }

            // Середня частина: від -0.85 до 0.85
            var midX = new List<double>();
            var midY = new List<double>();
            for (double x = -0.85; x < 0.85; x += 0.02)
            {
                double y = TargetFunction(x);
                if (!double.IsNaN(y) && Math.Abs(y) < 50)
                {
                    midX.Add(x);
                    midY.Add(y);
                }
            }

            // Права частина: від 1.15 до xMax
            var rightX = new List<double>();
            var rightY = new List<double>();
            for (double x = 1.15; x <= xMax; x += 0.02)
            {
                double y = TargetFunction(x);
                if (!double.IsNaN(y) && Math.Abs(y) < 50)
                {
                    rightX.Add(x);
                    rightY.Add(y);
                }
            }

            // Малюємо всі три частини
            bool firstSegment = true;
            if (leftX.Count > 0)
            {
                var lineLeft = plt.Add.Scatter(leftX.ToArray(), leftY.ToArray());
                lineLeft.Color = ScottPlot.Color.FromHex("#FF0000");
                lineLeft.LineWidth = 2;
                lineLeft.MarkerSize = 0;
                if (firstSegment)
                {
                    lineLeft.LegendText = "Цільова функція: y=(x-3)²+(x+1)/(x²-1)";
                    firstSegment = false;
                }
            }

            if (midX.Count > 0)
            {
                var lineMid = plt.Add.Scatter(midX.ToArray(), midY.ToArray());
                lineMid.Color = ScottPlot.Color.FromHex("#FF0000");
                lineMid.LineWidth = 2;
                lineMid.MarkerSize = 0;
            }

            if (rightX.Count > 0)
            {
                var lineRight = plt.Add.Scatter(rightX.ToArray(), rightY.ToArray());
                lineRight.Color = ScottPlot.Color.FromHex("#FF0000");
                lineRight.LineWidth = 2;
                lineRight.MarkerSize = 0;
            }

            plt.Title("Початкові дані (Варіант 2)");
            plt.XLabel("x");
            plt.YLabel("y");
            //plt.ShowLegend();
            plt.Legend.IsVisible = false;
            plt.Grid.IsVisible = true;

            plt.SavePng("initial_data.png", 800, 600);
        }

        // Візуалізація результатів передбачення
        static void VisualizeResults(ITransformer model, MLContext mlContext,
            List<RegressionInput> testData, double xMin, double xMax)
        {
            var plt = new ScottPlot.Plot();

            // Сортуємо тестові дані по X для коректного відображення
            var sorted = testData.OrderBy(d => d.X).ToList();
            double[] testX = sorted.Select(d => (double)d.X).ToArray();
            double[] testY = sorted.Select(d => (double)d.Y).ToArray();

            var realScatter = plt.Add.Scatter(testX, testY);
            realScatter.Color = ScottPlot.Color.FromHex("#0000FF");
            realScatter.MarkerSize = 8;
            realScatter.LineWidth = 0;
            realScatter.LegendText = "Реальні дані";

            // Передбачення моделі
            var predictionEngine = mlContext.Model.CreatePredictionEngine<RegressionInput, RegressionPrediction>(model);
            double[] predictedY = sorted.Select(d =>
                (double)predictionEngine.Predict(d).PredictedY).ToArray();

            var predLine = plt.Add.Scatter(testX, predictedY);
            predLine.Color = ScottPlot.Color.FromHex("#00FF00");
            predLine.MarkerSize = 0;
            predLine.LineWidth = 2;
            predLine.LegendText = "Передбачення моделі";

            // Ідеальна функція для порівняння (три сегменти)
            // Ліва частина
            var idealLeftX = new List<double>();
            var idealLeftY = new List<double>();
            for (double x = xMin; x < -1.15; x += 0.02)
            {
                double y = TargetFunction(x);
                if (!double.IsNaN(y) && Math.Abs(y) < 50)
                {
                    idealLeftX.Add(x);
                    idealLeftY.Add(y);
                }
            }

            // Середня частина
            var idealMidX = new List<double>();
            var idealMidY = new List<double>();
            for (double x = -0.85; x < 0.85; x += 0.02)
            {
                double y = TargetFunction(x);
                if (!double.IsNaN(y) && Math.Abs(y) < 50)
                {
                    idealMidX.Add(x);
                    idealMidY.Add(y);
                }
            }

            // Права частина
            var idealRightX = new List<double>();
            var idealRightY = new List<double>();
            for (double x = 1.15; x <= xMax; x += 0.02)
            {
                double y = TargetFunction(x);
                if (!double.IsNaN(y) && Math.Abs(y) < 50)
                {
                    idealRightX.Add(x);
                    idealRightY.Add(y);
                }
            }

            bool firstIdealSegment = true;
            if (idealLeftX.Count > 0)
            {
                var idealLineLeft = plt.Add.Scatter(idealLeftX.ToArray(), idealLeftY.ToArray());
                idealLineLeft.Color = ScottPlot.Color.FromHex("#FF0000");
                idealLineLeft.LineWidth = 1;
                idealLineLeft.LinePattern = ScottPlot.LinePattern.Dashed;
                idealLineLeft.MarkerSize = 0;
                if (firstIdealSegment)
                {
                    idealLineLeft.LegendText = "Цільова функція";
                    firstIdealSegment = false;
                }
            }

            if (idealMidX.Count > 0)
            {
                var idealLineMid = plt.Add.Scatter(idealMidX.ToArray(), idealMidY.ToArray());
                idealLineMid.Color = ScottPlot.Color.FromHex("#FF0000");
                idealLineMid.LineWidth = 1;
                idealLineMid.LinePattern = ScottPlot.LinePattern.Dashed;
                idealLineMid.MarkerSize = 0;
            }

            if (idealRightX.Count > 0)
            {
                var idealLineRight = plt.Add.Scatter(idealRightX.ToArray(), idealRightY.ToArray());
                idealLineRight.Color = ScottPlot.Color.FromHex("#FF0000");
                idealLineRight.LineWidth = 1;
                idealLineRight.LinePattern = ScottPlot.LinePattern.Dashed;
                idealLineRight.MarkerSize = 0;
            }

            plt.Title("Результати передбачення (Тестова вибірка)");
            plt.XLabel("x");
            plt.YLabel("y");
            //plt.ShowLegend();
            plt.Legend.IsVisible = false;
            plt.Grid.IsVisible = true;

            plt.SavePng("predictions.png", 800, 600);
        }

        // Інтерпретація коефіцієнта детермінації R²
        static void InterpretR2(double trainR2, double testR2)
        {
            Console.WriteLine("\n=== ІНТЕРПРЕТАЦІЯ ===");

            if (testR2 >= 0.9)
                Console.WriteLine("+ Відмінна якість моделі (R² ≥ 0.9)");
            else if (testR2 >= 0.7)
                Console.WriteLine("+ Хороша якість моделі (R² ≥ 0.7)");
            else if (testR2 >= 0.5)
                Console.WriteLine("! Задовільна якість моделі (R² ≥ 0.5)");
            else if (testR2 >= 0)
                Console.WriteLine("! Низька якість моделі (R² < 0.5)");
            else
                Console.WriteLine("!! Модель працює гірше за середнє значення (R² < 0)");

            // Перевірка на перенавчання
            double diff = trainR2 - testR2;
            if (diff > 0.1)
                Console.WriteLine($"!! Можливе перенавчання (різниця R²: {diff:F4})");
            else
                Console.WriteLine("+ Перенавчання відсутнє");
        }
    }
}