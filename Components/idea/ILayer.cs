using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Components
{
    public partial interface ILayer
    {
        public void Forward(ReadOnlySpan<float> input, Span<float> output);

        void Backward(ReadOnlySpan<float> outputGradient, Span<float> inputGradient, float learningRate);
    }
}
