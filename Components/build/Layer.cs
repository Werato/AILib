using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Components
{
    public class Layer 
    {
        public long NumberId { get; set; }//Id слоя 
        public double Byes {  get; set; }// byes (global)

        public int InDim, OutDim; // input output (Summators) (to-do: pass neirin ID for deep logging)
        public float[] W; // weights
        public float[] B;

        public Layer(int inDim, int outDim, Func<float> init)
        {
            InDim = inDim; OutDim = outDim;
            W = new float[outDim * inDim];
            for(int o = 0; o < outDim; o++)
            {
                int row = o * inDim;
                for (int i = 0; i < inDim; i++) 
                {
                    W[row + i] = init();// no better then Random 
                }
            }
        }

        // y = f(Wx + b)
        public void CreateLayer(List<float> x, List<float> y)//(ReadOnlySpan<float> x, Span<float> y)
        {
            int n = InDim;
            //new parallel threadS run OutDim times.
            //Lambda o is thread index
            System.Threading.Tasks.Parallel.For(0, OutDim, o =>
            {
                int row = o * n;
                //2 10 
                float sum = B[o];

                // SIMD-friendly dot product
                int i = 0;
                int vCount = Vector<float>.Count;
                var acc = Vector<float>.Zero;
                for (; i <= n - vCount; i += vCount)
                    acc += new Vector<float>(W, row + i) * new Vector<float>(x.Skip(i).ToArray());
                sum += Vector.Dot(acc, Vector<float>.One);
                for (; i < n; i++) sum += W[row + i] * x[i];

                ReadOnlySpan<float> s;
                //var aba = s.Slice(1);
                // activation (example: ReLU)
                y[o] = MathF.Max(0f, sum);
            });
        }

    }
}
