using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

public static class Numbers101
{
    // ---------- Data loading: MNIST CSV with 785 columns (label + 784 pixels) ----------
    // Use a CSV where the first value is label (0..9), next 784 are pixel intensities 0..255.
    // We'll keep only two labels (e.g., 0 and 1) for this first lesson.
    public static (List<float[]>, List<int>) LoadMnistCsvBinary(string path, int labelA, int labelB, int max = 60000)
    {
        var xs = new List<float[]>(capacity: Math.Min(max, 60000));
        var ys = new List<int>(capacity: Math.Min(max, 60000));

        using var sr = new StreamReader(path);
        string? line;
        int count = 0;

        // If your file has a header, skip it:
         var header = sr.ReadLine();

        while ((line = sr.ReadLine()) != null && count < max)
        {
            var parts = line.Split(',');
            if (parts.Length < 785) continue;//size of a picture

            int label = int.Parse(parts[0]); 
            if (label != labelA && label != labelB) continue;  //??

            // Convert pixels to [0,1]
            var x = new float[784];
            for (int i = 0; i < 784; i++)
            {
                float v = float.Parse(parts[i + 1]);
                x[i] = v / 255f;
            }

            // Map labels: labelA -> 0, labelB -> 1
            int y = (label == labelB) ? 1 : 0;

            xs.Add(x);
            ys.Add(y);
            count++;
        }
        return (xs, ys);
    }

    // ---------- Model: logistic regression (784 -> 1) ----------
    public sealed class Logistic784
    {
        public readonly int InputSize = 784;
        public float[] W; // weights[784]
        public float B;   // bias

        public Logistic784(int? seed = null)
        {
            W = new float[InputSize];
            var rng = seed.HasValue ? new Random(seed.Value) : new Random();

            // Xavier-like small init (not critical for logistic regression)
            float a = (float)Math.Sqrt(6.0 / (InputSize + 1));
            for (int i = 0; i < InputSize; i++)
            {
                float r = (float)(rng.NextDouble() * 2.0 - 1.0);
                W[i] = r * a * 0.1f; // extra small
            }
            B = 0f;
        }

        public float PredictProb(ReadOnlySpan<float> x)
        {
            float z = B;
            for (int i = 0; i < InputSize; i++)
                z += W[i] * x[i];

            // Sigmoid
            return 1f / (1f + MathF.Exp(-z));
        }

        public int PredictLabel(ReadOnlySpan<float> x)
        {
            return PredictProb(x) >= 0.5f ? 1 : 0;
        }

        // Train with mini-batches; returns average loss for monitoring
        public float TrainEpoch(IReadOnlyList<float[]> xs, IReadOnlyList<int> ys, int batchSize, float lr)
        {
            int n = xs.Count;
            if (n == 0) return 0f;

            float totalLoss = 0f;
            int batches = 0;

            // Simple SGD over data in order (you can shuffle if you want)
            for (int start = 0; start < n; start += batchSize)
            {
                int end = Math.Min(start + batchSize, n);
                int m = end - start;

                // Gradients
                float[] dW = new float[InputSize];
                float dB = 0f;

                float batchLoss = 0f;

                for (int idx = start; idx < end; idx++)
                {
                    var x = xs[idx];
                    int y = ys[idx];

                    float p = PredictProb(x);
                    // Numerically stable clamp
                    p = Math.Clamp(p, 1e-6f, 1f - 1e-6f);

                    // Loss
                    batchLoss += (float)(-(y * MathF.Log(p) + (1 - y) * MathF.Log(1 - p)));

                    // Gradients: (p - y) * x
                    float diff = p - y;
                    for (int i = 0; i < InputSize; i++)
                        dW[i] += diff * x[i];
                    dB += diff;
                }

                // Average gradients
                float invM = 1f / m;//////////////////
                for (int i = 0; i < InputSize; i++)
                    dW[i] *= invM;
                dB *= invM;
                batchLoss *= invM;

                // Update
                for (int i = 0; i < InputSize; i++)
                    W[i] -= lr * dW[i];
                B -= lr * dB;

                totalLoss += batchLoss;
                batches++;
            }

            return totalLoss / Math.Max(1, batches);
        }

        public float Accuracy(IReadOnlyList<float[]> xs, IReadOnlyList<int> ys)
        {
            int correct = 0;
            for (int i = 0; i < xs.Count; i++)
                if (PredictLabel(xs[i]) == ys[i]) correct++;
            return xs.Count == 0 ? 0f : (float)correct / xs.Count;
        }
    }

    // ---------- Demo runner ----------
    public static void RunBinaryDemo(string csvPath, int labelA = 0, int labelB = 1)
    {
        // 1) Load data (filter to two digits)
        var (xs, ys) = LoadMnistCsvBinary(csvPath, labelA, labelB, max: 12000);

        // Split train/test (simple split)
        int split = (int)(xs.Count * 0.8);
        var trainX = xs.GetRange(0, split);
        var trainY = ys.GetRange(0, split);
        var testX = xs.GetRange(split, xs.Count - split);
        var testY = ys.GetRange(split, ys.Count - split);

        // 2) Create model
        var model = new Logistic784(seed: 42);

        // 3) Train
        int epochs = 10;
        int batchSize = 64;
        float lr = 0.5f;  // start a bit high; if unstable, reduce to 0.1

        for (int e = 1; e <= epochs; e++)
        {
            float loss = model.TrainEpoch(trainX, trainY, batchSize, lr);
            float accTrain = model.Accuracy(trainX, trainY);
            float accTest = model.Accuracy(testX, testY);
            Console.WriteLine($"Epoch {e:00} | loss={loss:F4} | train={accTrain:P1} | test={accTest:P1}");
        }

        // 4) Example prediction (first test sample)
        if (testX.Count > 0)
        {
            int pred = model.PredictLabel(testX[0]);
            Console.WriteLine($"Example: pred={pred}, true={testY[0]}");
        }
    }
}
