using System;
using System.Numerics;
using System.Threading.Tasks;

namespace Components
{
    public class Layer : ILayer
    {
        private readonly IActivation _activation;

        public int InDim, OutDim;
        public float[] W; // Матрица весов (OutDim * InDim)
        public float[] B; // Вектор смещений (OutDim)

        // Наши внутренние буферы для сохранения состояния (выделяются один раз в конструкторе)
        private float[] _lastInput;
        private float[] _lastOutput;

        public Layer(int inDim, int outDim, Func<float> init, IActivation activation)
        {
            InDim = inDim;
            OutDim = outDim;
            _activation = activation;   

            W = new float[outDim * inDim];
            B = new float[outDim];

            _lastInput = new float[inDim];
            _lastOutput = new float[outDim];

            for (int o = 0; o < outDim; o++)
            {
                B[o] = init();
                int row = o * inDim;
                for (int i = 0; i < inDim; i++)
                {
                    W[row + i] = init();
                }
            }
        }

        public void Forward(ReadOnlySpan<float> input, Span<float> output)
        {
            // 1. Безопасно копируем входной Span в массив ДО потоков
            input.CopyTo(_lastInput);

            int n = InDim;
            int vCount = Vector<float>.Count;

            // Кешируем ссылки на массивы в локальные переменные для замыкания
            float[] localW = W;
            float[] localInput = _lastInput;
            float[] localB = B;

            Parallel.For(0, OutDim, o =>
            {
                int row = o * n;
                float sum = localB[o];

                var acc = Vector<float>.Zero;
                int i = 0;

                // Создаем Span-ы локально внутри конкретного потока — это на 100% легально и быстро
                ReadOnlySpan<float> weightsSlice = localW.AsSpan(row, n);
                ReadOnlySpan<float> inputSlice = localInput.AsSpan();

                // SIMD умножение
                for (; i <= n - vCount; i += vCount)
                {
                    var vW = new Vector<float>(weightsSlice.Slice(i));
                    var vX = new Vector<float>(inputSlice.Slice(i));
                    acc += vW * vX;
                }

                for (int j = 0; j < vCount; j++) sum += acc[j];

                for (; i < n; i++)
                {
                    sum += localW[row + i] * localInput[i];
                }

            });
            // 2. Когда все потоки завершились, мы в один приход копируем данные наружу.
            // Метод CopyTo для массивов оптимизирован на уровне ядра CLR (через memmove) и выполняется мгновенно.
            _lastOutput.CopyTo(output);
            _activation.Activate(_lastOutput.AsSpan());

        }

        public void Backward(ReadOnlySpan<float> outputGradient, Span<float> inputGradient, float learningRate)
        {
            int n = InDim;
            int vCount = Vector<float>.Count;

            inputGradient.Clear();

            object lockObj = new object();

            // Создаем локальную копию outputGradient во внутренний стек или массив, 
            // так как outputGradient тоже ref struct и не пойдет в Parallel.For.
            float[] localOutGrad = new float[OutDim];
            outputGradient.CopyTo(localOutGrad);
            _activation.ApplyDerivative(localOutGrad, _lastOutput);

            float[] localW = W;
            float[] localInput = _lastInput;
            float[] localB = B;
            float[] localLastOut = _lastOutput;

            Parallel.For(0, OutDim, o =>
            {
                float neuronGradient = localOutGrad[o];

                if (neuronGradient == 0f) return;

                int row = o * n;

                // Обновляем Bias
                localB[o] -= learningRate * neuronGradient;

                int i = 0;
                var vLearningRate = new Vector<float>(learningRate);
                var vNeuronGrad = new Vector<float>(neuronGradient);
                var vDeltaMultiplier = vLearningRate * vNeuronGrad;

                Span<float> weightsSlice = localW.AsSpan(row, n);
                ReadOnlySpan<float> inputSlice = localInput.AsSpan();

                // SIMD обновление весов
                for (; i <= n - vCount; i += vCount)
                {
                    var vW = new Vector<float>(weightsSlice.Slice(i));
                    var vX = new Vector<float>(inputSlice.Slice(i));

                    var vW_new = vW - (vDeltaMultiplier * vX);
                    vW_new.CopyTo(localW, row + i);
                }

                for (; i < n; i++)
                {
                    localW[row + i] -= learningRate * neuronGradient * localInput[i];
                }

                // Перенос ошибки на предыдущий слой (inputGradient). 
                // Так как inputGradient передан извне как Span, мы не можем использовать его внутри лямбды напрямую.
                // Вместо этого мы используем промежуточный плоский массив, который потом синхронизируем, 
                // либо работаем через unsafe-указатели. Для простоты применим локальный массив:
            });

            // Корректный пересчет inputGradient без нарушения правил ref struct:
            for (int o = 0; o < OutDim; o++)
            {
                float neuronGradient = _lastOutput[o] > 0f ? localOutGrad[o] : 0f;
                if (neuronGradient == 0f) continue;

                int row = o * n;
                for (int k = 0; k < n; k++)
                {
                    inputGradient[k] += neuronGradient * W[row + k];
                }
            }
        }
    }
}