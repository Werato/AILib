namespace Components
{
    public interface IActivation
    {
        /// <summary>
        /// Прямой проход. Применяет функцию активации к массиву значений (In-Place).
        /// </summary>
        void Activate(Span<float> values);

        /// <summary>
        /// Обратный проход. Умножает пришедшие градиенты на производную функции активации.
        /// </summary>
        /// <param name="gradients">Градиенты, которые нужно скорректировать (In-Place).</param>
        /// <param name="activatedOutputs">Уже активированные выходы этого слоя (сохраненные при Forward).</param>
        void ApplyDerivative(Span<float> gradients, ReadOnlySpan<float> activatedOutputs);
    }
}