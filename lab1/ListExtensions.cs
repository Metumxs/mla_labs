using System;
using System.Collections.Generic;

namespace lab1
{
    /// <summary>
    /// Розширення для списків — Fisher–Yates shuffle.
    /// Єдиний Random для застосунку.
    /// </summary>
    public static class ListExtensions
    {
        private static readonly Random rng = new Random();

        public static void Shuffle<T>(this IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                (list[k], list[n]) = (list[n], list[k]);
            }
        }
    }
}
