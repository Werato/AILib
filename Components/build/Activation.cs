using System;

namespace Components
{
    public class Relu : IActivation
    {
        public void Activate(Span<float> values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = MathF.Max(0f, values[i]);
            }
        }

        public void ApplyDerivative(Span<float> gradients, ReadOnlySpan<float> activatedOutputs)
        {
            for (int i = 0; i < gradients.Length; i++)
            {
                // Производная ReLU: 1, если выход > 0, иначе 0
                gradients[i] *= activatedOutputs[i] > 0f ? 1f : 0f;
            }
        }
    }

    public class Sigmoid : IActivation
    {
        public void Activate(Span<float> values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                // Sigmoid = 1 / (1 + e^-x)
                values[i] = 1f / (1f + MathF.Exp(-values[i]));
            }
        }

        public void ApplyDerivative(Span<float> gradients, ReadOnlySpan<float> activatedOutputs)
        {
            for (int i = 0; i < gradients.Length; i++)
            {
                // Производная Sigmoid: f(x) * (1 - f(x))
                float output = activatedOutputs[i];
                gradients[i] *= output * (1f - output);
            }
        }
    }
}