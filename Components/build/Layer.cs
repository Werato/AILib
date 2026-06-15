using System;
using System.Numerics;
using System.Threading.Tasks;

namespace Components
{
    public class Layer : ILayer
    {
        public long NumberId { get; set; } // Id слоя 
        public double Byes { get; set; } // оставим для совместимости

        public int InDim, OutDim;
        public float[] W; // Веса (размер: OutDim * InDim)
        public float[] B; // Смещения (размер: OutDim)

        // Кеш для хранения состояния между Forward и Backward (выделяется 1 раз)
        private float[] _lastInput;
        private float[] _lastOutput;

        public Layer(int inDim, int outDim, Func<float> init)
        {
            InDim = inDim;
            OutDim = outDim;

            W = new float[outDim * inDim];
            B = new float[outDim]; // Не забудь инициализировать массив B!

            _lastInput = new float[inDim];
            _lastOutput = new float[outDim];

            // Инициализация весов и смещений
            for (int o = 0; o < outDim; o++)
            {
                B[o] = init(); // Инициализируем bias небольшими случайными числами
                int row = o * inDim;
                for (int i = 0; i < inDim; i++)
                {
                    W[row + i] = init();
                }
            }
        }

        // Приводим к единому имени интерфейса (Forward вместо Forvard)
        public void Forward(ReadOnlySpan<float> input, Span<float> output)
        {
            // Сохраняем входные и выходные значения для последующего шага Backward
            input.CopyTo(_lastInput);

            int n = InDim;
            int vCount = Vector<float>.Count;

            Parallel.For(0, OutDim, o =>
            {
                int row = o * n;
                float sum = B[o];

                var acc = Vector<float>.Zero;
                int i = 0;

                ReadOnlySpan<float> weightsSlice = W.AsSpan(row, n);

                for (; i <= n - vCount; i += vCount)
                {
                    var vW = new Vector<float>(weightsSlice.Slice(i));
                    var vX = new Vector<float>(input.Slice(i));
                    acc += vW * vX;
                }

                for (int j = 0; j < vCount; j++) sum += acc[j];

                for (; i < n; i++)
                {
                    sum += W[row + i] * input[i];
                }

                // Активация ReLU
                output[o] = MathF.Max(0f, sum);
                _lastOutput[o] = output[o]; // Запоминаем выход
            });
        }

        public void Backward(ReadOnlySpan<float> outputGradient, Span<float> inputGradient, float learningRate)
        {
            int n = InDim;
            int vCount = Vector<float>.Count;

            // 1. Обнуляем входящий градиент для предыдущего слоя, так как мы будем аккумулировать его сумму
            inputGradient.Clear();

            // Будем использовать объект блокировки для потокобезопасного обновления inputGradient,
            // так как несколько потоков параллельно будут писать в одни и те же индексы inputGradient.
            // Альтернатива (более быстрая): сделать ThreadLocal буферы, но для старта это усложнит код.
            object lockObj = new object();

            Parallel.For(0, OutDim, o =>
            {
                // Производная функции активации ReLU:
                // Если выход слоя был <= 0, то градиент через этот нейрон не течет (равен 0)
                float neuronGradient = _lastOutput[o] > 0f ? outputGradient[o] : 0f;

                if (neuronGradient == 0f) return;

                int row = o * n;

                // Обновляем смещение (Bias) для текущего нейрона
                B[o] -= learningRate * neuronGradient;

                // Обновляем веса (W) текущего нейрона и считаем градиент для предыдущего слоя (inputGradient)
                int i = 0;

                // Переводим обновление весов на SIMD
                var vLearningRate = new Vector<float>(learningRate);
                var vNeuronGrad = new Vector<float>(neuronGradient);
                var vDeltaWeightMultiplier = vLearningRate * vNeuronGrad; // lr * grad

                Span<float> weightsSlice = W.AsSpan(row, n);
                ReadOnlySpan<float> lastInputSlice = _lastInput.AsSpan();

                for (; i <= n - vCount; i += vCount)
                {
                    // Обновление весов (SIMD): W = W - lr * grad * input
                    var vW = new Vector<float>(weightsSlice.Slice(i));
                    var vX = new Vector<float>(lastInputSlice.Slice(i));

                    var vW_new = vW - (vDeltaWeightMultiplier * vX);
                    vW_new.CopyTo(W, row + i);
                }

                // Досчитываем хвосты для весов
                for (; i < n; i++)
                {
                    W[row + i] -= learningRate * neuronGradient * _lastInput[i];
                }

                // Вычисляем градиент для предыдущего слоя (обратный проход через матрицу весов)
                // Ошибку нужно «раскидать» по всем входам, пропорционально весам.
                lock (lockObj)
                {
                    for (int k = 0; k < n; k++)
                    {
                        inputGradient[k] += neuronGradient * W[row + k];
                    }
                }
            });
        }
    }
}