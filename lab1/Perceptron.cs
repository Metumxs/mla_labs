using System;
using System.Collections.Generic;

namespace lab1
{
    /// <summary>
    /// Класичний перцептрон Розенблатта з 2 входами та bias.
    /// Має методи для епохового навчання і для оновлення по одному зразку.
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
            _learningRate = learningRate;
        }

        /// <summary>
        /// Обчислення прогнозу для однієї точки (0 або 1).
        /// </summary>
        public int Predict(DataPoint d)
        {
            double net = _w1 * d.Time + _w2 * d.Altitude + _b;
            return net >= 0 ? 1 : 0;
        }

        /// <summary>
        /// Оновлення ваг по одному прикладу (online update).
        /// Повертає true, якщо відбулося оновлення (delta != 0).
        /// </summary>
        public bool UpdateOneSample(DataPoint d)
        {
            int y = Predict(d);
            int delta = d.Label - y;
            if (delta != 0)
            {
                _w1 += _learningRate * delta * d.Time;
                _w2 += _learningRate * delta * d.Altitude;
                _b += _learningRate * delta;
                return true;
            }
            return false;
        }

        /// <summary>
        /// Одна епоха навчання (повний прохід по списку).
        /// </summary>
        public void TrainOneEpoch(List<DataPoint> trainingData, out int errorsCount)
        {
            errorsCount = 0;
            foreach (var d in trainingData)
            {
                int y = Predict(d);
                int delta = d.Label - y;
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

        /// <summary>
        /// (Опціонально) встановити ваги вручну — корисно для повторних експериментів.
        /// </summary>
        public void SetWeights(double w1, double w2, double b)
        {
            _w1 = w1; _w2 = w2; _b = b;
        }
    }
}
