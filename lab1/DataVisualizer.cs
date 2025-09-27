using System.Collections.Generic;
using System.Linq;
using ScottPlot; // Простір імен той самий

namespace lab1
{
    public static class DataVisualizer
    {
        public static void PlotData(
            List<DataPoint> allData,
            (double w1, double w2, double b) weights,
            double minTime, double maxTime, double minAlt, double maxAlt,
            string filePath = "drone_classification_plot.png")
        {
            // 1. Створення графіка (конструктор тепер без аргументів)
            var plt = new Plot();

            // 2. Розділення даних за класами
            var amateurDrones = allData.Where(d => d.Type == DroneType.Amateur).ToList();
            var professionalDrones = allData.Where(d => d.Type == DroneType.Professional).ToList();

            double[] amateurTimes = amateurDrones.Select(d => d.Time).ToArray();
            double[] amateurAlts = amateurDrones.Select(d => d.Altitude).ToArray();
            double[] profTimes = professionalDrones.Select(d => d.Time).ToArray();
            double[] profAlts = professionalDrones.Select(d => d.Altitude).ToArray();

            // 3. Додавання точок (методи тепер викликаються через plt.Add)
            //    Використовуємо кольори з ScottPlot, щоб уникнути конфлікту
            var amateurScatter = plt.Add.Scatter(amateurTimes, amateurAlts);
            amateurScatter.Label = "Amateur";
            amateurScatter.Color = Colors.Blue;
            amateurScatter.MarkerSize = 5;
            amateurScatter.LineWidth = 0;

            var profScatter = plt.Add.Scatter(profTimes, profAlts);
            profScatter.Label = "Professional";
            profScatter.Color = Colors.Red;
            profScatter.MarkerSize = 5;
            profScatter.LineWidth = 0;

            // 4. Розрахунок та побудова роздільної лінії
            if (weights.w2 != 0)
            {
                double x1_norm = 0.0;
                double y1_norm = (-weights.w1 * x1_norm - weights.b) / weights.w2;

                double x2_norm = 1.0;
                double y2_norm = (-weights.w1 * x2_norm - weights.b) / weights.w2;

                double rangeTime = maxTime - minTime;
                double rangeAlt = maxAlt - minAlt;

                double lineX1 = x1_norm * rangeTime + minTime;
                double lineY1 = y1_norm * rangeAlt + minAlt;

                double lineX2 = x2_norm * rangeTime + minTime;
                double lineY2 = y2_norm * rangeAlt + minAlt;

                // Метод AddLine тепер також через plt.Add
                var decisionLine = plt.Add.Line(lineX1, lineY1, lineX2, lineY2);
                decisionLine.Label = "Decision Boundary";
                decisionLine.Color = Colors.Green;
                decisionLine.LineWidth = 2;
            }

            // 5. Налаштування графіка
            plt.Title("Drone Classification");
            plt.XLabel("Time (minutes)");
            plt.YLabel("Altitude (meters)");

            // Легенда тепер є властивістю, а не методом
            plt.Legend.IsVisible = true;
            plt.Legend.Location = Alignment.LowerRight;

            // AxisAuto() більше не потрібен, осі налаштовуються автоматично

            // 6. Збереження графіка (метод SaveFig замінено на Save)
            plt.Save(filePath, 600, 400);
        }
    }
}