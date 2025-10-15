using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace lab2_2
{
    // Клас для зберігання вхідних даних (дві ознаки)
    public class HumidityInput
    {
        [LoadColumn(0)]
        public float Temperature { get; set; }  // Температура (°C)

        [LoadColumn(1)]
        public float WindSpeed { get; set; }    // Швидкість вітру (м/с)

        [LoadColumn(2)]
        public float Humidity { get; set; }     // Вологість (%)
    }

    // Клас для зберігання нормалізованих даних
    public class NormalizedHumidityInput
    {
        public float Temperature { get; set; }
        public float WindSpeed { get; set; }
        public float Humidity { get; set; }

        // Для зворотного перетворення
        public float OriginalTemperature { get; set; }
        public float OriginalWindSpeed { get; set; }
    }

    // Клас для передбачень
    public class HumidityPrediction
    {
        [ColumnName("Score")]
        public float PredictedHumidity { get; set; }
    }

    // Клас для зберігання результатів експерименту
    public class ExperimentResult
    {
        public int ExperimentNumber { get; set; }
        public double TrainTestRatio { get; set; }
        public string NoiseType { get; set; }
        public double NoiseLevel { get; set; }
        public double TrainR2 { get; set; }
        public double TestR2 { get; set; }
        public double MAE { get; set; }
        public double RMSE { get; set; }
    }

    class Program
    {
        // Параметри для нормалізації
        static float tempMin, tempMax, windMin, windMax;

        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.WriteLine("=== Лабораторна робота 2: Завдання 2 ===");
            Console.WriteLine("Варіант 2: Передбачення рівня вологості");
            Console.WriteLine("Ознаки: Температура (°C) та Швидкість вітру (м/с)\n");

            // Список для збереження результатів експериментів
            var experimentResults = new List<ExperimentResult>();

            // Експеримент 1: Базовий (80/20, нормальний шум 3.0)
            Console.WriteLine("\n" + new string('=', 70));
            Console.WriteLine("ЕКСПЕРИМЕНТ 1: Базовий (80/20, нормальний шум σ=3.0)");
            Console.WriteLine(new string('=', 70));
            var result1 = RunExperiment(1, 0.80, "нормальний", 3.0);
            experimentResults.Add(result1);

            // Експеримент 2: Більше навчальних даних (85/15)
            Console.WriteLine("\n" + new string('=', 70));
            Console.WriteLine("ЕКСПЕРИМЕНТ 2: Більше навчальних даних (85/15, нормальний шум σ=3.0)");
            Console.WriteLine(new string('=', 70));
            var result2 = RunExperiment(2, 0.85, "нормальний", 3.0);
            experimentResults.Add(result2);

            // Експеримент 3: Менший шум (80/20, шум 1.5)
            Console.WriteLine("\n" + new string('=', 70));
            Console.WriteLine("ЕКСПЕРИМЕНТ 3: Менший шум (80/20, нормальний шум σ=1.5)");
            Console.WriteLine(new string('=', 70));
            var result3 = RunExperiment(3, 0.80, "нормальний", 1.5);
            experimentResults.Add(result3);

            // Експеримент 4: Більший шум (80/20, шум 5.0)
            Console.WriteLine("\n" + new string('=', 70));
            Console.WriteLine("ЕКСПЕРИМЕНТ 4: Більший шум (80/20, нормальний шум σ=5.0)");
            Console.WriteLine(new string('=', 70));
            var result4 = RunExperiment(4, 0.80, "нормальний", 5.0);
            experimentResults.Add(result4);

            // Експеримент 5: Рівномірний шум (80/20, рівномірний шум ±5)
            Console.WriteLine("\n" + new string('=', 70));
            Console.WriteLine("ЕКСПЕРИМЕНТ 5: Рівномірний шум (80/20, рівномірний шум ±5)");
            Console.WriteLine(new string('=', 70));
            var result5 = RunExperiment(5, 0.80, "рівномірний", 5.0);
            experimentResults.Add(result5);

            // Виведення зведеної таблиці результатів
            Console.WriteLine("\n" + new string('=', 70));
            Console.WriteLine("ЗВЕДЕНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ");
            Console.WriteLine(new string('=', 70));
            PrintResultsTable(experimentResults);

            // Аналіз результатів
            Console.WriteLine("\n" + new string('=', 70));
            Console.WriteLine("АНАЛІЗ РЕЗУЛЬТАТІВ");
            Console.WriteLine(new string('=', 70));
            AnalyzeResults(experimentResults);

            Console.WriteLine("\n✓ Всі експерименти виконано!");
        }

        // Функція для виконання одного експерименту
        static ExperimentResult RunExperiment(int expNumber, double trainRatio, string noiseType, double noiseLevel)
        {
            Console.WriteLine($"\n[1] Генерація 200 синтетичних даних...");

            // Діапазони ознак
            double tempMinRange = -10.0;  // °C
            double tempMaxRange = 40.0;   // °C
            double windMinRange = 0.0;    // м/с
            double windMaxRange = 20.0;   // м/с

            Console.WriteLine($"   Діапазон температури: [{tempMinRange}, {tempMaxRange}] °C");
            Console.WriteLine($"   Діапазон швидкості вітру: [{windMinRange}, {windMaxRange}] м/с");

            // Генерація початкових даних
            var rawData = GenerateSyntheticData(200, tempMinRange, tempMaxRange,
                windMinRange, windMaxRange, noiseType, noiseLevel, expNumber);

            Console.WriteLine($"\n[2] Масштабування ознак (Min-Max нормалізація)...");
            var normalizedData = NormalizeData(rawData);
            Console.WriteLine($"   Температура нормалізована: [{tempMin:F2}, {tempMax:F2}] → [0, 1]");
            Console.WriteLine($"   Швидкість вітру нормалізована: [{windMin:F2}, {windMax:F2}] → [0, 1]");

            Console.WriteLine($"\n[3] Розділення на навчальну ({trainRatio * 100:F0}%) та тестову ({(1 - trainRatio) * 100:F0}%) вибірки...");
            var (trainData, testData) = SplitData(normalizedData, trainRatio, expNumber);
            Console.WriteLine($"   Навчальна вибірка: {trainData.Count} записів");
            Console.WriteLine($"   Тестова вибірка: {testData.Count} записів");

            Console.WriteLine($"\n[4] Створення моделі регресії...");
            Console.WriteLine("   Алгоритм: LightGBM (Light Gradient Boosting Machine)");
            Console.WriteLine("   Кількість дерев: 100");
            Console.WriteLine("   Кількість листків: 31");
            Console.WriteLine("   Learning rate: 0.1");

            var mlContext = new MLContext(seed: 42 + expNumber);

            // Перетворення даних у формат ML.NET
            var trainInputs = trainData.Select(d => new HumidityInput
            {
                Temperature = d.Temperature,
                WindSpeed = d.WindSpeed,
                Humidity = d.Humidity
            }).ToList();

            var testInputs = testData.Select(d => new HumidityInput
            {
                Temperature = d.Temperature,
                WindSpeed = d.WindSpeed,
                Humidity = d.Humidity
            }).ToList();

            var trainDataView = mlContext.Data.LoadFromEnumerable(trainInputs);
            var testDataView = mlContext.Data.LoadFromEnumerable(testInputs);

            // Створення pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features",
                    nameof(HumidityInput.Temperature),
                    nameof(HumidityInput.WindSpeed))
                .Append(mlContext.Regression.Trainers.LightGbm(
                    labelColumnName: nameof(HumidityInput.Humidity),
                    featureColumnName: "Features",
                    numberOfLeaves: 31,
                    numberOfIterations: 100,
                    learningRate: 0.1,
                    minimumExampleCountPerLeaf: 5));

            Console.WriteLine($"\n[5] Навчання моделі...");
            var model = pipeline.Fit(trainDataView);
            Console.WriteLine("   ✓ Навчання завершено");

            Console.WriteLine($"\n[6] Оцінка якості моделі...");

            // Оцінка на навчальній вибірці
            var trainPredictions = model.Transform(trainDataView);
            var trainMetrics = mlContext.Regression.Evaluate(trainPredictions,
                labelColumnName: nameof(HumidityInput.Humidity));

            // Оцінка на тестовій вибірці
            var testPredictions = model.Transform(testDataView);
            var testMetrics = mlContext.Regression.Evaluate(testPredictions,
                labelColumnName: nameof(HumidityInput.Humidity));

            Console.WriteLine("\n=== РЕЗУЛЬТАТИ ЕКСПЕРИМЕНТУ ===");
            Console.WriteLine($"R² (навчальна вибірка): {trainMetrics.RSquared:F4}");
            Console.WriteLine($"R² (тестова вибірка):   {testMetrics.RSquared:F4}");
            Console.WriteLine($"MAE (тестова):          {testMetrics.MeanAbsoluteError:F4}");
            Console.WriteLine($"RMSE (тестова):         {testMetrics.RootMeanSquaredError:F4}");

            InterpretR2(trainMetrics.RSquared, testMetrics.RSquared);

            return new ExperimentResult
            {
                ExperimentNumber = expNumber,
                TrainTestRatio = trainRatio,
                NoiseType = noiseType,
                NoiseLevel = noiseLevel,
                TrainR2 = trainMetrics.RSquared,
                TestR2 = testMetrics.RSquared,
                MAE = testMetrics.MeanAbsoluteError,
                RMSE = testMetrics.RootMeanSquaredError
            };
        }

        // Генерація синтетичних даних
        static List<HumidityInput> GenerateSyntheticData(int count,
            double tempMin, double tempMax,
            double windMin, double windMax,
            string noiseType, double noiseLevel, int seed)
        {
            var random = new Random(42 + seed);
            var data = new List<HumidityInput>();

            for (int i = 0; i < count; i++)
            {
                // Генерація випадкових ознак
                double temperature = tempMin + (tempMax - tempMin) * random.NextDouble();
                double windSpeed = windMin + (windMax - windMin) * random.NextDouble();

                // Формула для обчислення вологості (фізично обґрунтована):
                // - При вищій температурі повітря може містити більше води, але відносна вологість падає
                // - Вітер збільшує випаровування, знижуючи вологість
                // - Базова вологість: 70%
                double baseHumidity = 70.0;
                double tempEffect = -0.4 * temperature;  // Температура знижує відносну вологість
                double windEffect = -1.2 * windSpeed;    // Вітер знижує вологість

                double idealHumidity = baseHumidity + tempEffect + windEffect;

                // Додавання шуму
                double noise = 0;
                if (noiseType == "нормальний")
                {
                    noise = NormalRandom(random) * noiseLevel;
                }
                else if (noiseType == "рівномірний")
                {
                    noise = (random.NextDouble() * 2 - 1) * noiseLevel;  // від -noiseLevel до +noiseLevel
                }

                double humidity = idealHumidity + noise;

                // Обмеження вологості в реалістичних межах (0-100%)
                humidity = Math.Max(0, Math.Min(100, humidity));

                data.Add(new HumidityInput
                {
                    Temperature = (float)temperature,
                    WindSpeed = (float)windSpeed,
                    Humidity = (float)humidity
                });
            }

            return data;
        }

        // Генерація нормально розподіленої випадкової величини
        static double NormalRandom(Random random)
        {
            double u1 = random.NextDouble();
            double u2 = random.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        // Нормалізація даних (Min-Max)
        static List<NormalizedHumidityInput> NormalizeData(List<HumidityInput> data)
        {
            // Знаходження мінімальних і максимальних значень
            tempMin = data.Min(d => d.Temperature);
            tempMax = data.Max(d => d.Temperature);
            windMin = data.Min(d => d.WindSpeed);
            windMax = data.Max(d => d.WindSpeed);

            var normalized = new List<NormalizedHumidityInput>();

            foreach (var item in data)
            {
                float normTemp = (item.Temperature - tempMin) / (tempMax - tempMin);
                float normWind = (item.WindSpeed - windMin) / (windMax - windMin);

                normalized.Add(new NormalizedHumidityInput
                {
                    Temperature = normTemp,
                    WindSpeed = normWind,
                    Humidity = item.Humidity,  // Цільову змінну не нормалізуємо
                    OriginalTemperature = item.Temperature,
                    OriginalWindSpeed = item.WindSpeed
                });
            }

            return normalized;
        }

        // Розділення даних на навчальну та тестову вибірки
        static (List<NormalizedHumidityInput> train, List<NormalizedHumidityInput> test)
            SplitData(List<NormalizedHumidityInput> data, double trainRatio, int seed)
        {
            var random = new Random(42 + seed);
            var shuffled = data.OrderBy(x => random.Next()).ToList();

            int trainCount = (int)(shuffled.Count * trainRatio);
            var train = shuffled.Take(trainCount).ToList();
            var test = shuffled.Skip(trainCount).ToList();

            return (train, test);
        }

        // Виведення таблиці результатів
        static void PrintResultsTable(List<ExperimentResult> results)
        {
            Console.WriteLine("\n┌─────┬───────────┬──────────────────┬───────┬──────────┬─────────┬────────┬────────┐");
            Console.WriteLine("│ №   │ Розподіл  │ Тип шуму         │ Рівень│ Train R² │ Test R² │  MAE   │  RMSE  │");
            Console.WriteLine("├─────┼───────────┼──────────────────┼───────┼──────────┼─────────┼────────┼────────┤");

            foreach (var result in results)
            {
                string ratio = $"{result.TrainTestRatio * 100:F0}/{(1 - result.TrainTestRatio) * 100:F0}";
                string noiseInfo = result.NoiseType == "нормальний"
                    ? $"норм. σ={result.NoiseLevel:F1}"
                    : $"рівном. ±{result.NoiseLevel:F1}";

                Console.WriteLine($"│ {result.ExperimentNumber,-3} │ {ratio,-9} │ {noiseInfo,-16} │ {result.NoiseLevel,5:F1} │ {result.TrainR2,8:F4} │ {result.TestR2,7:F4} │ {result.MAE,6:F3} │ {result.RMSE,6:F3} │");
            }

            Console.WriteLine("└─────┴───────────┴──────────────────┴───────┴──────────┴─────────┴────────┴────────┘");
        }

        // Аналіз результатів
        static void AnalyzeResults(List<ExperimentResult> results)
        {
            Console.WriteLine("\n1. ВПЛИВ РОЗПОДІЛУ ДАНИХ:");
            var exp1 = results[0];  // 80/20
            var exp2 = results[1];  // 85/15
            Console.WriteLine($"   > При збільшенні навчальної вибірки (80→85%):");
            Console.WriteLine($"     Train R²: {exp1.TrainR2:F4} → {exp2.TrainR2:F4} (Δ = {exp2.TrainR2 - exp1.TrainR2:+F4})");
            Console.WriteLine($"     Test R²:  {exp1.TestR2:F4} → {exp2.TestR2:F4} (Δ = {exp2.TestR2 - exp1.TestR2:+F4})");

            if (exp2.TestR2 > exp1.TestR2)
                Console.WriteLine("     ✓ Більше навчальних даних покращило якість моделі");
            else
                Console.WriteLine("     ! Збільшення навчальних даних не дало значного покращення");

            Console.WriteLine("\n2. ВПЛИВ РІВНЯ ШУМУ (нормальний розподіл):");
            var exp3 = results[2];  // шум 1.5
            var exp4 = results[3];  // шум 5.0
            Console.WriteLine($"   > Малий шум (σ=1.5):   Test R² = {exp3.TestR2:F4}, RMSE = {exp3.RMSE:F3}");
            Console.WriteLine($"   > Середній шум (σ=3.0): Test R² = {exp1.TestR2:F4}, RMSE = {exp1.RMSE:F3}");
            Console.WriteLine($"   > Великий шум (σ=5.0):  Test R² = {exp4.TestR2:F4}, RMSE = {exp4.RMSE:F3}");
            Console.WriteLine("   Висновок: Зі збільшенням шуму якість моделі погіршується (нижче R², вище RMSE)");

            Console.WriteLine("\n3. ВПЛИВ ТИПУ ШУМУ:");
            var exp5 = results[4];  // рівномірний шум
            Console.WriteLine($"   > Нормальний шум:    Test R² = {exp1.TestR2:F4}, RMSE = {exp1.RMSE:F3}");
            Console.WriteLine($"   > Рівномірний шум:   Test R² = {exp5.TestR2:F4}, RMSE = {exp5.RMSE:F3}");

            if (Math.Abs(exp1.TestR2 - exp5.TestR2) < 0.05)
                Console.WriteLine("   Висновок: Тип шуму має незначний вплив при однаковому рівні");
            else if (exp1.TestR2 > exp5.TestR2)
                Console.WriteLine("   Висновок: Модель краще справляється з нормальним шумом");
            else
                Console.WriteLine("   Висновок: Модель краще справляється з рівномірним шумом");

            Console.WriteLine("\n4. ПЕРЕНАВЧАННЯ:");
            foreach (var result in results)
            {
                double diff = result.TrainR2 - result.TestR2;
                string status = diff > 0.1 ? "! Можливе перенавчання" : "✓ Перенавчання відсутнє";
                Console.WriteLine($"   Експеримент {result.ExperimentNumber}: Δ(R²) = {diff:F4}  {status}");
            }

            Console.WriteLine("\n5. ЗАГАЛЬНИЙ ВИСНОВОК:");
            var bestExp = results.OrderByDescending(r => r.TestR2).First();
            Console.WriteLine($"   Найкраща модель: Експеримент {bestExp.ExperimentNumber}");
            Console.WriteLine($"   > Розподіл: {bestExp.TrainTestRatio * 100:F0}/{(1 - bestExp.TrainTestRatio) * 100:F0}");
            Console.WriteLine($"   > Шум: {bestExp.NoiseType}, рівень {bestExp.NoiseLevel:F1}");
            Console.WriteLine($"   > Test R² = {bestExp.TestR2:F4}");
            Console.WriteLine("\n   Рекомендації:");
            Console.WriteLine("   > Використовувати мінімальний рівень шуму для максимальної точності");
            Console.WriteLine("   > Збільшувати навчальну вибірку для стабільніших результатів");
            Console.WriteLine("   > Модель LightGBM добре справляється з нелінійними залежностями");
        }

        // Інтерпретація R²
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

            double diff = trainR2 - testR2;
            if (diff > 0.1)
                Console.WriteLine($"! Можливе перенавчання (різниця R²: {diff:F4})");
            else
                Console.WriteLine("+ Перенавчання відсутнє");
        }
    }
}