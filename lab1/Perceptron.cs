using System;
using System.Collections.Generic;

namespace lab1
{
    /// <summary>
    /// Класичний перцептрон Розенблатта з 2 входами та bias.
    /// </summary>
    public class Perceptron
    {
        private double _w1, _w2, _b;
        private readonly double _learningRate;
        private readonly Random _rnd;

        public Perceptron(double learningRate = 0.1, int? randomSeed = null)
        {
            _rnd = randomSeed.HasValue ? new Random(randomSeed.Value) : new Random();
            _w1 = _rnd.NextDouble() * 0.1 - 0.05; // в межах [-0.05, 0.05)
            _w2 = _rnd.NextDouble() * 0.1 - 0.05;
            _b = _rnd.NextDouble() * 0.1 - 0.05;
            this._learningRate = learningRate;
        }

        /// <summary>
        /// Повертає 0 або 1 — прогноз для одного прикладу (великі числа — очікувано нормалізовані).
        /// </summary>
        public int Predict(DataPoint d)
        {
            double net = _w1 * d.Time + _w2 * d.Altitude + _b;
            return net >= 0 ? 1 : 0;
        }

        /// <summary>
        /// Одна епоха навчання.
        /// </summary>
        public void TrainOneEpoch(List<DataPoint> trainingData, out int errorsCount)
        {
            errorsCount = 0;
            foreach (var d in trainingData)
            {
                int y = Predict(d);
                int delta = d.Label - y; // Label = (int)Type

                if (delta != 0)
                {
                    _w1 += _learningRate * delta * d.Time;
                    _w2 += _learningRate * delta * d.Altitude;
                    _b += _learningRate * delta;
                    ++errorsCount;
                }
            }
        }

        public (double w1, double w2, double b) GetWeights() => (_w1, _w2, _b);
    }
}
