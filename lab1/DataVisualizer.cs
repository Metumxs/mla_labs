using System.Collections.Generic;
using System.Linq;
using ScottPlot;

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
            var plt = new Plot();

            var amateurDrones = allData.Where(d => d.Type == DroneType.Amateur).ToList();
            var professionalDrones = allData.Where(d => d.Type == DroneType.Professional).ToList();

            double[] amateurTimes = amateurDrones.Select(d => d.Time).ToArray();
            double[] amateurAlts = amateurDrones.Select(d => d.Altitude).ToArray();
            double[] profTimes = professionalDrones.Select(d => d.Time).ToArray();
            double[] profAlts = professionalDrones.Select(d => d.Altitude).ToArray();

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

            if (weights.w2 != 0)
            {
                // Визначення діапазону для побудови лінії
                double rangeTime = maxTime - minTime;
                double rangeAlt = maxAlt - minAlt;

                // Розрахунок координат початку та кінця лінії
                double lineX1 = minTime;
                double x1_norm = (lineX1 - minTime) / rangeTime;
                double y1_norm = (-weights.w1 * x1_norm - weights.b) / weights.w2;
                double lineY1 = y1_norm * rangeAlt + minAlt;

                double lineX2 = maxTime;
                double x2_norm = (lineX2 - minTime) / rangeTime;
                double y2_norm = (-weights.w1 * x2_norm - weights.b) / weights.w2;
                double lineY2 = y2_norm * rangeAlt + minAlt;

                // Побудова лінії між цими точками
                var decisionLine = plt.Add.Line(lineX1, lineY1, lineX2, lineY2);
                decisionLine.Label = "Decision Boundary";
                decisionLine.Color = Colors.Green;
                decisionLine.LineWidth = 2;
            }

            plt.Title("Drone Classification");
            plt.XLabel("Time (minutes)");
            plt.YLabel("Altitude (meters)");
            plt.Legend.IsVisible = true;
            plt.Legend.Location = Alignment.LowerRight;

            plt.Save(filePath, 600, 400);
        }
    }
}