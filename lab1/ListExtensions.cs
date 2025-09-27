namespace lab1
{
    /// <summary>
    /// Клас для методів розширення.
    /// </summary>
    public static class ListExtensions
    {
        // Створюємо один екземпляр Random для всього застосунку,
        // щоб уникнути проблем з однаковими послідовностями.
        private static readonly Random rng = new Random();

        /// <summary>
        /// Перемішує елементи списку у випадковому порядку (алгоритм Фішера-Єтса).
        /// </summary>
        public static void Shuffle<T>(this IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                // Обмін елементами
                (list[k], list[n]) = (list[n], list[k]);
            }
        }
    }
}