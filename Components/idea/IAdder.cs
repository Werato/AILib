using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Components
{
    public partial interface IAdder
    {
        /// <summary>
        /// Вектор весов нейрона. Используем Span для быстрой работы с памятью без аллокаций.
        /// </summary>
        public Span<float> Weights { get; }

        /// <summary>
        /// Смещение (Bias) нейрона. Обязательный элемент для сдвига функции активации.
        /// </summary>
        public float Bias { get; set; }

        /// <summary>
        /// Вычисляет взвешенную сумму и пропускает её через функцию активации.
        /// </summary>
        /// <param name="inputs">Входные сигналы, пришедшие на этот нейрон (только для чтения).</param>
        /// <returns>Выходное значение нейрона (0..1 или иные границы в зависимости от функции).</returns>
        public float Activate(ReadOnlySpan<float> inputs);
    }
}
