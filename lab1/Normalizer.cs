using System.Collections.Generic;
using System.Linq;

namespace lab1
{
    /// <summary>
    /// Реалізація min-max нормалізації для датасету.
    /// Якщо max == min для якоїсь ознаки, то значення цієї ознаки встановлюються в 0.0 (щоб уникнути ділення на нуль).
    /// </summary>
    public static class Normalizer
    {
        public static void MinMaxNormalize(List<DataPoint> data, out double minTime, out double maxTime, out double minAlt, out double maxAlt)
        {
            minTime = data.Min(d => d.Time);
            maxTime = data.Max(d => d.Time);
            minAlt = data.Min(d => d.Altitude);
            maxAlt = data.Max(d => d.Altitude);

            double rangeTime = maxTime - minTime;
            double rangeAlt = maxAlt - minAlt;

            foreach (var d in data)
            {
                d.Time = rangeTime == 0 ? 0.0 : (d.Time - minTime) / rangeTime;
                d.Altitude = rangeAlt == 0 ? 0.0 : (d.Altitude - minAlt) / rangeAlt;
            }
        }

        public static void ApplyNormalization(List<DataPoint> data, double minTime, double maxTime, double minAlt, double maxAlt)
        {
            double rangeTime = maxTime - minTime;
            double rangeAlt = maxAlt - minAlt;

            foreach (var d in data)
            {
                d.Time = rangeTime == 0 ? 0.0 : (d.Time - minTime) / rangeTime;
                d.Altitude = rangeAlt == 0 ? 0.0 : (d.Altitude - minAlt) / rangeAlt;
            }
        }
    }
}
