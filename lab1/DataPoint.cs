namespace lab1
{
    /// <summary>
    /// Представляє одну точку даних (дрон) з ID, часом роботи, висотою та типом (enum).
    /// Id — захищений (private set), Type може змінюватися.
    /// Time та Altitude — змінні (наприклад для нормалізації).
    /// </summary>
    public class DataPoint
    {
        public int Id { get; private set; }
        public DroneType Type { get; set; } // тепер змінюваний

        public double Time { get; set; }     // може змінюватися після нормалізації
        public double Altitude { get; set; } // може змінюватися після нормалізації

        // Зручне числове представлення типу для навчання (0 або 1)
        public int Label => (int)Type;

        public DataPoint(int id, double time, double altitude, DroneType type)
        {
            Id = id;
            Time = time;
            Altitude = altitude;
            Type = type;
        }

        public override string ToString()
        {
            return $"ID={Id}: ({Time:F2}, {Altitude:F2}), Type={Type}";
        }
    }
}





