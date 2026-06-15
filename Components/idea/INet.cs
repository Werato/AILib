using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Components
{
    public partial interface INet
    {
        /// <summary>
        /// Список всех слоев нейросети в порядке их выполнения.
        /// </summary>
        public IEnumerable<ILayer> Layers { get; }

        /// <summary>
        /// Прямой проход по всей сети (вычисление предсказания).
        /// </summary>
        /// <param name="input">Входной вектор признаков (например, пиксели картинки).</param>
        /// <param name="output">Буфер, куда запишется итоговый результат работы сети (вероятности классов).</param>
        void Predict(ReadOnlySpan<float> input, Span<float> output);

        /// <summary>
        /// Одна итерация обучения сети (Forward + Backward) на одном примере.
        /// </summary>
        /// <param name="input">Входной вектор признаков.</param>
        /// <param name="target">Целевое (ожидаемое) значение, с которым сравнивается выход сети.</param>
        /// <param name="learningRate">Скорость обучения (шаг градиентного спуска).</param>
        /// <returns>Значение ошибки (Loss) на данном примере (для мониторинга обучения).</returns>
        float Train(ReadOnlySpan<float> input, ReadOnlySpan<float> target, float learningRate);
    }
}
