using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Components
{
    // Класс-DTO для удобного свертывания/развертывания сети (в JSON или файл)
    public class NetSnapshotDto
    {
        public List<LayerSnapshotDto> Layers { get; set; } = new();
    }

    public class LayerSnapshotDto
    {
        public int InDim { get; set; }
        public int OutDim { get; set; }
        public float[] Weights { get; set; }
        public float[] Biases { get; set; }
    }

    public class Net : INet
    {
        private readonly List<Layer> _layers = new();

        // Экспонируем слои как ILayer через интерфейс
        public IEnumerable<ILayer> Layers => _layers;

        // Промежуточные буферы для обмена данными между слоями (ноль аллокаций в рантайме)
        private float[][] _forwardBuffers;
        private float[][] _backwardBuffers;

        public Net() { }

        // Конструктор инициализации новой сети
        public Net(int inputDim, List<int> hiddenLayers, int outputDim, Func<float> init, IActivation activation)
        {
            int currentInDim = inputDim;

            // Строим цепочку слоев
            foreach (int outDim in hiddenLayers)
            {
                _layers.Add(new Layer(currentInDim, outDim, init, activation));
                currentInDim = outDim;
            }
            _layers.Add(new Layer(currentInDim, outputDim, init, activation));

            AllocateInternalBuffers(inputDim);
        }

        // Выделение памяти под промежуточные буферы (вызывается 1 раз при старте или загрузке)
        private void AllocateInternalBuffers(int inputDim)
        {
            int layerCount = _layers.Count;
            _forwardBuffers = new float[layerCount - 1][];
            _backwardBuffers = new float[layerCount - 1][];

            for (int i = 0; i < layerCount - 1; i++)
            {
                int bufferSize = _layers[i].OutDim;
                _forwardBuffers[i] = new float[bufferSize];
                _backwardBuffers[i] = new float[bufferSize];
            }
        }

        public void Predict(ReadOnlySpan<float> input, Span<float> output)
        {
            if (_layers.Count() == 0) return;

            // Если слой всего один
            if (_layers.Count() == 1)
            {
                _layers[0].Forward(input, output);
                return;
            }

            // Прямой проход сквозь цепочку буферов
            _layers[0].Forward(input, _forwardBuffers[0]);

            for (int i = 1; i < _layers.Count() - 1; i++)
            {
                _layers[i].Forward(_forwardBuffers[i - 1], _forwardBuffers[i]);
            }

            _layers[^1].Forward(_forwardBuffers[^1], output);
        }

        public float Train(ReadOnlySpan<float> input, ReadOnlySpan<float> target, float learningRate)
        {
            // 1. Выделяем временное место под предсказание сети (конечный выход)
            float[] prediction = new float[_layers[^1].OutDim];
            Predict(input, prediction);

            // 2. Считаем ошибку (Loss) и градиент на выходе
            // Для простоты используем Mean Squared Error (MSE) на верхнем уровне
            float[] outputGradient = new float[prediction.Length];
            float loss = 0f;

            for (int i = 0; i < prediction.Length; i++)
            {
                float error = prediction[i] - target[i];
                loss += error * error;
                outputGradient[i] = error; // Производная от MSE ошибки
            }
            loss /= prediction.Length;

            // 3. Обратный проход (Backpropagation) от конца к началу
            if (_layers.Count == 1)
            {
                float[] dummyGrad = new float[_layers[0].InDim];
                _layers[0].Backward(outputGradient, dummyGrad, learningRate);
                return loss;
            }

            // Обратный шаг для последнего слоя
            _layers[^1].Backward(outputGradient, _backwardBuffers[^1], learningRate);

            // Обратный шаг для скрытых слоев
            for (int i = _layers.Count - 2; i > 0; i--)
            {
                _layers[i].Backward(_backwardBuffers[i], _backwardBuffers[i - 1], learningRate);
            }

            // Обратный шаг для первого слоя
            float[] firstLayerInputGrad = new float[_layers[0].InDim]; // нам не нужен градиент для самого входа, но метод требует буфер
            _layers[0].Backward(_backwardBuffers[0], firstLayerInputGrad, learningRate);

            return loss;
        }

        #region СВЕРТЫВАНИЕ И РАЗВЕРТЫВАНИЕ (Save / Load)

        /// <summary>
        /// Сворачивает текущую сеть в JSON строку. Можно сохранить в файл или БД.
        /// </summary>
        public string SaveToJson()
        {
            var snapshot = new NetSnapshotDto();
            foreach (var layer in _layers)
            {
                snapshot.Layers.Add(new LayerSnapshotDto
                {
                    InDim = layer.InDim,
                    OutDim = layer.OutDim,
                    Weights = layer.W.ToArray(), // Копируем массивы
                    Biases = layer.B.ToArray()
                });
            }

            return JsonSerializer.Serialize(snapshot, new JsonSerializerOptions { WriteIndented = true });
        }

        /// <summary>
        /// Разворачивает сеть из JSON строки, восстанавливая все веса.
        /// </summary>
        public static Net LoadFromJson(string json)
        {
            var snapshot = JsonSerializer.Deserialize<NetSnapshotDto>(json);
            if (snapshot == null || snapshot.Layers.Count == 0)
                throw new ArgumentException("Некорректный формат JSON для нейросети.");

            var net = new Net();

            // Восстанавливаем слои из сохраненных весов
            foreach (var layerDto in snapshot.Layers)
            {
                // Передаем заглушку-инициализатор, так как веса мы перезапишем сразу ниже
                var layer = new Layer(layerDto.InDim, layerDto.OutDim, () => 0f);

                Array.Copy(layerDto.Weights, layer.W, layerDto.Weights.Length);
                Array.Copy(layerDto.Biases, layer.B, layerDto.Biases.Length);

                net._layers.Add(layer);
            }

            // Инициализируем внутренние буферы обмена на основе восстановленной геометрии
            net.AllocateInternalBuffers(net._layers[0].InDim);

            return net;
        }

        #endregion
    }
}